"""Compare live/paper performance against the backtest, flagging significant
divergence — an early warning for model or market-regime drift. Feeds the
retraining trigger documented in docs/runbook.md.

Reconstructs round-trip trades from the structured fill log
(`src/execution/structured_logging.py`) via simple per-symbol FIFO pairing
of buy/sell fills, then reuses `backtest.metrics.compute_backtest_metrics`
for a like-for-like comparison against the backtest report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backtest.metrics import compute_backtest_metrics


def load_fills(log_path: Path) -> pd.DataFrame:
    records = []
    with log_path.open() as f:
        for line in f:
            record = json.loads(line)
            if record.get("event_type") == "fill":
                records.append(record)
    if not records:
        return pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price", "order_id"])
    return pd.DataFrame(records)


def reconstruct_trades_from_fills(fills: pd.DataFrame) -> pd.DataFrame:
    """FIFO-pair buy/sell fills per symbol into round-trip trades.

    `holding_bars` isn't meaningful here (fills carry wall-clock time, not
    bar counts) — set to NaN rather than a misleading bar-count guess, which
    just makes `avg_trade_duration_bars` NaN in the resulting metrics.
    """
    trades = []
    for symbol, group in fills.sort_values("timestamp").groupby("symbol"):
        open_buys: list[dict] = []
        for _, fill in group.iterrows():
            if fill["side"] == "buy":
                open_buys.append(
                    {"price": fill["price"], "qty": fill["qty"], "timestamp": fill["timestamp"]}
                )
            elif fill["side"] == "sell":
                remaining = fill["qty"]
                while remaining > 0 and open_buys:
                    buy = open_buys[0]
                    matched_qty = min(remaining, buy["qty"])
                    pnl = (fill["price"] - buy["price"]) * matched_qty
                    trades.append(
                        {
                            "symbol": symbol,
                            "entry_time": buy["timestamp"],
                            "exit_time": fill["timestamp"],
                            "entry_price": buy["price"],
                            "exit_price": fill["price"],
                            "shares": matched_qty,
                            "pnl": pnl,
                            "return_pct": pnl / (buy["price"] * matched_qty),
                            "holding_bars": float("nan"),
                        }
                    )
                    buy["qty"] -= matched_qty
                    remaining -= matched_qty
                    if buy["qty"] <= 0:
                        open_buys.pop(0)
    return pd.DataFrame(trades)


def build_daily_equity_from_trades(trades: pd.DataFrame, starting_equity: float) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=float)
    trades = trades.sort_values("exit_time")
    exit_dates = pd.to_datetime(trades["exit_time"]).dt.date
    cumulative_pnl = trades.groupby(exit_dates)["pnl"].sum().cumsum()
    return starting_equity + cumulative_pnl


@dataclass
class DivergenceFlag:
    metric: str
    live_value: float
    baseline_value: float
    message: str


def compare_to_baseline(
    live_metrics: dict, baseline_metrics: dict, tolerance: float = 0.5
) -> list[DivergenceFlag]:
    """Flag metrics whose relative gap from baseline exceeds `tolerance`."""
    flags = []
    for key in ["win_rate", "expectancy", "sharpe", "max_drawdown"]:
        live_val, base_val = live_metrics.get(key), baseline_metrics.get(key)
        if live_val is None or base_val is None or base_val != base_val or live_val != live_val:
            continue
        if base_val == 0:
            continue
        relative_gap = abs(live_val - base_val) / abs(base_val)
        if relative_gap > tolerance:
            flags.append(
                DivergenceFlag(
                    metric=key,
                    live_value=live_val,
                    baseline_value=base_val,
                    message=(
                        f"{key}: live={live_val:.4f} vs baseline={base_val:.4f} "
                        f"({relative_gap:.0%} relative gap, threshold {tolerance:.0%})"
                    ),
                )
            )
    return flags


def build_live_metrics(log_path: Path, starting_equity: float) -> dict:
    fills = load_fills(log_path)
    trades = reconstruct_trades_from_fills(fills)
    daily_equity = build_daily_equity_from_trades(trades, starting_equity)
    return compute_backtest_metrics(trades, daily_equity)
