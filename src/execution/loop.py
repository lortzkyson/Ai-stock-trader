"""Single iteration of the paper-trading loop.

Meant to be invoked on a schedule (e.g. every 5 minutes during regular
trading hours) by an external scheduler — this module does not loop or
sleep itself; each call is one complete iteration: check the kill switch,
manage existing tracked positions (stop/target/max-holding exits), check
for new entry signals across the universe, submit orders, and persist state
for the next invocation.

Data feed: live bars come from Alpaca's free real-time IEX feed
(`DataFeed.IEX`), not the SIP feed training/backtesting use — a real,
documented mismatch (see docs/go_live_review.md), accepted for now since
the model doesn't have a demonstrated edge regardless of feed. This loop
exists to build the paper-trading operational track record and prove the
plumbing, not to be read as strategy validation until that's resolved.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.trading.enums import OrderSide

from data.alpaca_client import AlpacaBarsClient, BarsQuery
from data.quality import filter_regular_session, find_duplicate_timestamps, find_impossible_values
from execution.client import AlpacaExecutionClient
from execution.kill_switch import KILL_SWITCH_FLAG, is_engaged
from execution.position_state import (
    DEFAULT_POSITION_STATE_PATH,
    load_open_positions,
    save_open_positions,
)
from execution.reconciliation import reconcile_positions
from execution.signal import compute_live_signal, load_production_model
from execution.structured_logging import DEFAULT_LOG_PATH, StructuredLogger
from risk.engine import RiskConfig, RiskEngine
from risk.state_persistence import DEFAULT_STATE_PATH, load_risk_state, save_risk_state

SESSION_START = "09:30"
_ET = ZoneInfo("America/New_York")


@dataclass
class LoopContext:
    exec_client: AlpacaExecutionClient
    bars_client: AlpacaBarsClient
    risk_engine: RiskEngine
    logger: StructuredLogger
    today: date


def _recent_trading_calendar(n: int = 10) -> list[date]:
    days: list[date] = []
    d = date.today()
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    return sorted(days)


def _fetch_live_bars(client: AlpacaBarsClient, symbol: str, lookback_minutes: int) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=lookback_minutes)
    query = BarsQuery(symbols=[symbol], start=start, end=end, feed=DataFeed.IEX)
    raw = client.fetch_bars(query)
    if raw.empty:
        return raw

    # Duplicate/impossible-value checks and the regular-session filter apply
    # fine to any window size, but check_quality's halt-like-day detection
    # assumes a full session's worth of bars to judge a gap against — a
    # short live lookback (e.g. 5 minutes for an exit check) would always
    # look like a near-total "halt" relative to a 390-minute session and get
    # wrongly stripped out entirely. Skip that check here; it's Phase 2's
    # data-pipeline concern for historical pulls, not this loop's.
    keep = ~find_duplicate_timestamps(raw) & ~find_impossible_values(raw)
    cleaned = raw.loc[keep].reset_index(drop=True)
    return filter_regular_session(cleaned)


def _fetch_session_bars(client: AlpacaBarsClient, symbol: str) -> pd.DataFrame:
    """Fetch from today's session open through now — required for the PO3/AMD
    features (accumulation range, sweeps), which are anchored to session open,
    not a rolling lookback window. A rolling window would silently miss the
    opening range any time after mid-morning and break those features.
    """
    now_et = datetime.now(timezone.utc).astimezone(_ET)
    hour, minute = (int(x) for x in SESSION_START.split(":"))
    session_open = now_et.replace(hour=hour, minute=minute, second=0, microsecond=0)
    end = datetime.now(timezone.utc)
    query = BarsQuery(symbols=[symbol], start=session_open, end=end, feed=DataFeed.IEX)
    raw = client.fetch_bars(query)
    if raw.empty:
        return raw

    keep = ~find_duplicate_timestamps(raw) & ~find_impossible_values(raw)
    cleaned = raw.loc[keep].reset_index(drop=True)
    return filter_regular_session(cleaned)


def run_iteration(
    universe: list[str],
    exec_client: AlpacaExecutionClient | None = None,
    bars_client: AlpacaBarsClient | None = None,
    model: Any = None,
    risk_config: RiskConfig | None = None,
    probability_threshold: float = 0.5,
    risk_state_path: Path = DEFAULT_STATE_PATH,
    position_state_path: Path = DEFAULT_POSITION_STATE_PATH,
    kill_switch_path: Path = KILL_SWITCH_FLAG,
    log_path: Path = DEFAULT_LOG_PATH,
) -> None:
    logger = StructuredLogger(log_path)
    risk_config = risk_config or RiskConfig()

    if is_engaged(kill_switch_path):
        logger.log("loop_skipped", reason="kill_switch_engaged")
        return

    exec_client = exec_client or AlpacaExecutionClient()
    bars_client = bars_client or AlpacaBarsClient()
    model = model if model is not None else load_production_model()

    if not exec_client.is_market_open():
        logger.log("loop_skipped", reason="market_closed")
        return

    risk_engine = RiskEngine(risk_config)
    load_risk_state(risk_engine, risk_state_path)

    equity = exec_client.get_account_equity()
    today = date.today()
    risk_engine.update_equity(today, equity)
    save_risk_state(risk_engine, risk_state_path)

    ctx = LoopContext(exec_client, bars_client, risk_engine, logger, today)

    open_positions = load_open_positions(position_state_path)
    actual_positions = {p.symbol: p for p in exec_client.get_positions()}
    expected = {
        sym: int(float(actual_positions[sym].qty)) if sym in actual_positions else 0
        for sym in open_positions
    }
    reconciliation = reconcile_positions(expected, exec_client, logger)
    if not reconciliation.is_consistent:
        for symbol in list(open_positions):
            if symbol not in actual_positions:
                del open_positions[symbol]  # closed outside our tracking; already logged above
        save_open_positions(open_positions, position_state_path)

    if risk_engine.drawdown_breaker.is_tripped:
        logger.log("loop_skipped", reason="max_drawdown_circuit_breaker_tripped")
        return
    if risk_engine.daily_loss_limit.is_tripped:
        logger.log("loop_skipped", reason="daily_loss_limit_tripped")
        return

    trading_calendar = _recent_trading_calendar()

    _manage_exits(ctx, open_positions, actual_positions)
    save_open_positions(open_positions, position_state_path)
    save_risk_state(risk_engine, risk_state_path)

    _check_entries(
        ctx, universe, open_positions, model, probability_threshold, equity, trading_calendar
    )
    save_open_positions(open_positions, position_state_path)
    save_risk_state(risk_engine, risk_state_path)

    # A healthy quiet run (nothing traded) must still leave a trace — otherwise
    # it's indistinguishable in the log from a run that crashed immediately.
    logger.log(
        "loop_completed", symbols_checked=len(universe), equity=equity,
        open_positions=len(open_positions),
    )


def _manage_exits(
    ctx: LoopContext, open_positions: dict[str, dict[str, Any]], actual_positions: dict[str, Any]
) -> None:
    for symbol in list(open_positions):
        if symbol not in actual_positions:
            continue  # already dropped above after reconciliation

        bars = _fetch_live_bars(ctx.bars_client, symbol, lookback_minutes=5)
        if bars.empty:
            continue
        latest = bars.iloc[-1]
        pos = open_positions[symbol]

        exit_price, exit_reason = None, None
        if float(latest["low"]) <= pos["stop_loss_price"]:
            exit_price, exit_reason = pos["stop_loss_price"], "stop_loss"
        elif float(latest["high"]) >= pos["take_profit_price"]:
            exit_price, exit_reason = pos["take_profit_price"], "take_profit"
        elif datetime.now(timezone.utc) >= datetime.fromisoformat(pos["max_holding_deadline"]):
            exit_price, exit_reason = float(latest["close"]), "max_holding"

        if exit_reason is None:
            continue

        qty = int(float(actual_positions[symbol].qty))
        client_order_id = f"exit-{symbol}-{uuid.uuid4().hex[:6]}"
        order = ctx.exec_client.submit_market_order(
            symbol, qty=qty, side=OrderSide.SELL, client_order_id=client_order_id
        )
        ctx.logger.log_order_submitted(
            symbol, side="sell", qty=qty, order_type="market", order_id=str(order.id)
        )
        ctx.logger.log("exit_signal", symbol=symbol, exit_reason=exit_reason, exit_price=exit_price)

        if pos["entry_date"] == ctx.today.isoformat():
            ctx.risk_engine.day_trade_tracker.record_day_trade(ctx.today)

        del open_positions[symbol]


def _check_entries(
    ctx: LoopContext,
    universe: list[str],
    open_positions: dict[str, dict[str, Any]],
    model: Any,
    probability_threshold: float,
    equity: float,
    trading_calendar: list[date],
) -> None:
    for symbol in universe:
        if symbol in open_positions:
            continue

        bars = _fetch_session_bars(ctx.bars_client, symbol)
        if len(bars) < 60:
            continue

        predicted, proba = compute_live_signal(bars, model, probability_threshold)
        ctx.logger.log_signal(symbol, proba=proba, predicted=predicted)
        if predicted != 1:
            continue

        entry_price_estimate = float(bars.iloc[-1]["close"])
        # Re-read gross exposure from the broker each iteration rather than
        # tracking it locally: it must reflect fills that actually happened,
        # including any this loop opened moments ago.
        gross_exposure = sum(
            abs(float(p.market_value)) for p in ctx.exec_client.get_positions()
        )
        decision = ctx.risk_engine.evaluate_entry(
            as_of_date=ctx.today, entry_price=entry_price_estimate, equity=equity,
            trading_calendar=trading_calendar, is_intended_day_trade=True,
            current_gross_exposure=gross_exposure,
        )
        if not decision.approved:
            ctx.logger.log(
                "entry_rejected", symbol=symbol, reason=decision.reason,
                gross_exposure=gross_exposure, equity=equity,
            )
            continue

        client_order_id = f"entry-{symbol}-{uuid.uuid4().hex[:6]}"
        order = ctx.exec_client.submit_market_order(
            symbol, qty=decision.shares, side=OrderSide.BUY, client_order_id=client_order_id
        )
        ctx.logger.log_order_submitted(
            symbol, side="buy", qty=decision.shares, order_type="market", order_id=str(order.id)
        )

        max_holding_minutes = ctx.risk_engine.config.barrier_config.max_holding_bars
        deadline = datetime.now(timezone.utc) + timedelta(minutes=max_holding_minutes)
        open_positions[symbol] = {
            "entry_price": entry_price_estimate,
            "stop_loss_price": decision.stop_loss_price,
            "take_profit_price": decision.take_profit_price,
            "entry_date": ctx.today.isoformat(),
            "max_holding_deadline": deadline.isoformat(),
        }
