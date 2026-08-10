"""Backtest performance metrics computed from the trade log and daily equity curve.

This is the authoritative performance measure for the system (unlike
src/models/metrics.py's lightweight per-trade proxy used during Phase 4
model comparison) — it reflects real fills, costs, and one shared pool of
capital across positions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def compute_backtest_metrics(trades: pd.DataFrame, daily_equity: pd.Series) -> dict:
    if trades.empty:
        return {
            "n_trades": 0,
            "win_rate": float("nan"),
            "expectancy": float("nan"),
            "profit_factor": float("nan"),
            "sharpe": float("nan"),
            "sortino": float("nan"),
            "max_drawdown": float("nan"),
            "avg_trade_duration_bars": float("nan"),
            "total_return": float("nan"),
        }

    win_rate = float((trades["pnl"] > 0).mean())
    expectancy = float(trades["return_pct"].mean())

    gross_profit = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gross_loss = -trades.loc[trades["pnl"] < 0, "pnl"].sum()
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    daily_returns = daily_equity.pct_change().dropna()
    sharpe = _sharpe(daily_returns)
    sortino = _sortino(daily_returns)

    running_max = daily_equity.cummax()
    drawdown = daily_equity / running_max - 1
    max_drawdown = float(drawdown.min()) if len(drawdown) else float("nan")

    total_return = (
        float(daily_equity.iloc[-1] / daily_equity.iloc[0] - 1)
        if len(daily_equity)
        else float("nan")
    )

    return {
        "n_trades": int(len(trades)),
        "win_rate": win_rate,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "avg_trade_duration_bars": float(trades["holding_bars"].mean()),
        "total_return": total_return,
    }


def _sharpe(daily_returns: pd.Series) -> float:
    if len(daily_returns) < 2 or daily_returns.std() == 0:
        return float("nan")
    return float(daily_returns.mean() / daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def _sortino(daily_returns: pd.Series) -> float:
    downside = daily_returns[daily_returns < 0]
    if len(daily_returns) < 2 or len(downside) == 0 or downside.std() == 0:
        return float("nan")
    return float(daily_returns.mean() / downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
