"""Signal-level metrics computed from triple-barrier `realized_return` outcomes.

This is a lightweight proxy backtest, not the realistic, cost-aware
event-driven backtester built in Phase 5 — it exists so Phase 4 can compare
model/threshold choices quickly without needing Phase 5's full engine. The
Phase 5 backtester is the authoritative performance number.

Sharpe/drawdown are computed from a *daily* equal-weighted return series
(mean of that day's trade returns), not by compounding individual trades
sequentially: with thousands of overlapping trades across 8 symbols, many
entered on the same day or with overlapping triple-barrier windows, naively
multiplying (1 + r) through every trade in sequence treats concurrent trades
as if they happened one after another with 100% of capital reinvested each
time — that compounds toward zero almost immediately and produced nonsense
(Sharpe > 100 alongside a -100% max drawdown) before this was caught. Daily
aggregation is still a simplification — Phase 5's backtester is the one that
actually respects position limits and capital — but it doesn't manufacture a
fictional blowup the way sequential per-trade compounding did.

Target metric is expectancy and Sharpe, not win rate alone (docs/pre-mortem.md
guard #7) — every metrics dict here reports win_rate purely as a diagnostic
alongside expectancy, never as the thing to optimize.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def compute_signal_metrics(
    realized_returns: np.ndarray,
    hit_labels: np.ndarray,
    entry_dates: np.ndarray,
) -> dict:
    if len(realized_returns) == 0:
        return {
            "n_trades": 0,
            "win_rate": float("nan"),
            "expectancy": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
        }

    win_rate = float((hit_labels == 1).mean())
    expectancy = float(realized_returns.mean())

    daily = pd.Series(realized_returns, index=pd.Index(entry_dates, name="date"))
    daily_mean_return = daily.groupby(level="date").mean()
    daily_std = daily_mean_return.std()
    sharpe = (
        float(daily_mean_return.mean() / daily_std * np.sqrt(TRADING_DAYS_PER_YEAR))
        if daily_std > 0
        else float("nan")
    )

    equity_curve = (1 + daily_mean_return).cumprod()
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve / running_max - 1
    max_drawdown = float(drawdown.min()) if len(drawdown) else float("nan")

    return {
        "n_trades": int(len(realized_returns)),
        "win_rate": win_rate,
        "expectancy": expectancy,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }
