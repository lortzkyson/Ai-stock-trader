"""Event-driven backtester. See docs/pre-mortem.md guards #5, #6.

Replays model signals against real historical bars from multiple symbols,
interleaved in chronological order and sharing one pool of capital (position
sizing reflects *account* equity, not a separate pool per symbol — that's
the whole point of routing sizing through src/risk/engine.py rather than
hardcoding it here).

Execution assumptions:
- fills happen no earlier than the bar after the signal (mandatory
  next-bar-open, plus `FillConfig.latency_bars` further delay)
- market vs. limit orders (src/backtest/fills.py), including the
  possibility a limit entry never fills
- within-bar stop/target ambiguity resolved conservatively (stop wins ties),
  matching src/features/labeling.py so backtest fills and training labels
  use the same rule
- Alpaca's real fee schedule (src/backtest/costs.py)
- every entry is sized and approved by src/risk/engine.py — this module
  has no position-sizing or risk logic of its own
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from backtest.costs import commission, sell_regulatory_fees
from backtest.fills import FillConfig, apply_exit_slippage, simulate_entry_fill
from risk.engine import RiskEngine


@dataclass
class OpenPosition:
    symbol: str
    entry_time: pd.Timestamp
    entry_date: date
    entry_price: float
    shares: int
    stop_loss_price: float
    take_profit_price: float
    max_holding_bars: int
    bars_held: int = 0


@dataclass
class PendingEntry:
    fill_at_bar_count: int  # this symbol's own bar counter value at which to attempt the fill


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    daily_equity: pd.Series


def _build_merged_stream(bars_by_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for symbol, bars in bars_by_symbol.items():
        df = bars.copy()
        df["symbol"] = symbol
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    return merged.sort_values(["timestamp", "symbol"], kind="stable").reset_index(drop=True)


def run_backtest(
    bars_by_symbol: dict[str, pd.DataFrame],
    signals_by_symbol: dict[str, pd.Series],
    risk_engine: RiskEngine,
    fill_config: FillConfig,
    starting_equity: float,
    trading_calendar: list[date],
) -> BacktestResult:
    merged = _build_merged_stream(bars_by_symbol)

    cash = starting_equity
    open_positions: dict[str, OpenPosition] = {}
    pending_entries: dict[str, PendingEntry] = {}
    bar_counts: dict[str, int] = dict.fromkeys(bars_by_symbol, 0)
    last_close: dict[str, float] = {}

    trades: list[dict] = []
    daily_equity: dict[date, float] = {}

    signal_lookup = {
        symbol: dict(zip(bars_by_symbol[symbol]["timestamp"], signals_by_symbol[symbol]))
        for symbol in bars_by_symbol
    }

    for row_raw in merged.itertuples(index=False):
        # pandas-stubs types itertuples rows too loosely to be useful here;
        # treat as Any and cast explicitly at each use instead.
        row: Any = row_raw
        symbol: str = str(row.symbol)
        row_timestamp = pd.Timestamp(row.timestamp)
        bar_counts[symbol] += 1
        last_close[symbol] = float(row.close)
        current_date = row_timestamp.date()

        # 1. Manage an existing open position: check for stop/target/max-holding exit.
        if symbol in open_positions:
            position = open_positions[symbol]
            position.bars_held += 1
            exit_price, exit_reason = _check_exit(row, position, fill_config)
            if exit_price is not None and exit_reason is not None:
                _close_position(
                    position, exit_price, exit_reason, row_timestamp,
                    trades, risk_engine, current_date,
                )
                fees = sell_regulatory_fees(position.shares, exit_price)
                cash += position.shares * exit_price - fees
                del open_positions[symbol]

        # 2. Fill a pending entry scheduled for this bar.
        if symbol in pending_entries and symbol not in open_positions:
            pending = pending_entries[symbol]
            if bar_counts[symbol] >= pending.fill_at_bar_count:
                del pending_entries[symbol]
                fill_price = simulate_entry_fill(pd.Series(row._asdict()), fill_config)
                if fill_price is not None:
                    equity_estimate = _mark_to_market(cash, open_positions, last_close)
                    decision = risk_engine.evaluate_entry(
                        as_of_date=current_date,
                        entry_price=fill_price,
                        equity=equity_estimate,
                        trading_calendar=trading_calendar,
                        # Exits here are triggered by stop/target/max-holding, not a
                        # pre-planned same-day close, so we can't know in advance
                        # whether a given entry will end up being a day trade.
                        # Conservative simplification: treat every entry as a
                        # potential day trade for PDT gating, which blocks *all*
                        # new entries (not just same-day ones) once the rolling
                        # budget is spent. docs/pre-mortem.md's more sophisticated
                        # design — keep allowing new entries and defer the *exit*
                        # of any position that would otherwise close same-day past
                        # the PDT limit — isn't implemented; that would mean
                        # holding a position open after its stop/target already
                        # triggered, which changes realized P&L and needs its own
                        # careful design. This is a documented known limitation,
                        # not a silent gap: it's strictly safer (never violates
                        # PDT) at the cost of being more conservative than
                        # necessary about which entries it blocks.
                        is_intended_day_trade=True,
                    )
                    if decision.approved:
                        order_commission = commission(decision.shares, fill_price)
                        cost = decision.shares * fill_price + order_commission
                        if cost <= cash:
                            cash -= cost
                            open_positions[symbol] = OpenPosition(
                                symbol=symbol,
                                entry_time=row_timestamp,
                                entry_date=current_date,
                                entry_price=fill_price,
                                shares=decision.shares,
                                stop_loss_price=decision.stop_loss_price,
                                take_profit_price=decision.take_profit_price,
                                max_holding_bars=risk_engine.config.barrier_config.max_holding_bars,
                            )

        # 3. New signal on this bar: schedule an entry for the future.
        signal = signal_lookup[symbol].get(row_timestamp, 0)
        if signal == 1 and symbol not in open_positions and symbol not in pending_entries:
            pending_entries[symbol] = PendingEntry(
                fill_at_bar_count=bar_counts[symbol] + 1 + fill_config.latency_bars
            )

        equity_now = _mark_to_market(cash, open_positions, last_close)
        risk_engine.update_equity(current_date, equity_now)
        daily_equity[current_date] = equity_now

    # Liquidate anything still open at the end of the data at the last known close.
    for symbol, position in list(open_positions.items()):
        exit_price = last_close.get(symbol, position.entry_price)
        exit_price = apply_exit_slippage(exit_price, fill_config.slippage_bps)
        _close_position(
            position, exit_price, "end_of_data", pd.Timestamp(merged["timestamp"].iloc[-1]),
            trades, risk_engine, pd.Timestamp(merged["timestamp"].iloc[-1]).date(),
        )
        cash += position.shares * exit_price - sell_regulatory_fees(position.shares, exit_price)

    trades_df = pd.DataFrame(trades)
    equity_series = pd.Series(daily_equity).sort_index()
    return BacktestResult(trades=trades_df, daily_equity=equity_series)


def _check_exit(
    row, position: OpenPosition, fill_config: FillConfig
) -> tuple[float | None, str | None]:
    high, low = float(row.high), float(row.low)
    profit_hit = high >= position.take_profit_price
    stop_hit = low <= position.stop_loss_price
    slippage = fill_config.slippage_bps

    if stop_hit:  # conservative tie-break: stop wins if both triggered in the same bar
        return apply_exit_slippage(position.stop_loss_price, slippage), "stop_loss"
    if profit_hit:
        return apply_exit_slippage(position.take_profit_price, slippage), "take_profit"
    if position.bars_held >= position.max_holding_bars:
        return apply_exit_slippage(float(row.close), slippage), "max_holding"
    return None, None


def _close_position(
    position: OpenPosition,
    exit_price: float,
    exit_reason: str,
    exit_time: pd.Timestamp,
    trades: list[dict],
    risk_engine: RiskEngine,
    exit_date: date,
) -> None:
    fees = sell_regulatory_fees(position.shares, exit_price)
    gross_pnl = (exit_price - position.entry_price) * position.shares
    pnl = gross_pnl - fees
    is_day_trade = position.entry_date == exit_date

    trades.append(
        {
            "symbol": position.symbol,
            "entry_time": position.entry_time,
            "exit_time": exit_time,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "shares": position.shares,
            "exit_reason": exit_reason,
            "pnl": pnl,
            "fees": fees,
            "return_pct": pnl / (position.entry_price * position.shares),
            "holding_bars": position.bars_held,
            "is_day_trade": is_day_trade,
        }
    )
    if is_day_trade:
        risk_engine.day_trade_tracker.record_day_trade(position.entry_date)


def _mark_to_market(
    cash: float, open_positions: dict[str, OpenPosition], last_close: dict[str, float]
) -> float:
    positions_value = sum(
        pos.shares * last_close.get(pos.symbol, pos.entry_price) for pos in open_positions.values()
    )
    return cash + positions_value
