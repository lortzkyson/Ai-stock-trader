from __future__ import annotations

import pandas as pd
import pytest

from backtest.metrics import compute_backtest_metrics


def test_compute_backtest_metrics_basic() -> None:
    trades = pd.DataFrame(
        {
            "pnl": [100.0, -50.0, 200.0, -50.0],
            "return_pct": [0.01, -0.005, 0.02, -0.005],
            "holding_bars": [10, 5, 20, 8],
        }
    )
    daily_equity = pd.Series(
        [100_000, 100_100, 100_050, 100_250, 100_200],
        index=pd.date_range("2026-01-05", periods=5, freq="D"),
    )

    metrics = compute_backtest_metrics(trades, daily_equity)

    assert metrics["n_trades"] == 4
    assert metrics["win_rate"] == pytest.approx(0.5)
    assert metrics["profit_factor"] == pytest.approx(300.0 / 100.0)
    assert metrics["avg_trade_duration_bars"] == pytest.approx((10 + 5 + 20 + 8) / 4)
    assert metrics["max_drawdown"] <= 0
    assert metrics["total_return"] == pytest.approx(100_200 / 100_000 - 1)


def test_compute_backtest_metrics_empty_trades() -> None:
    metrics = compute_backtest_metrics(pd.DataFrame(), pd.Series(dtype=float))
    assert metrics["n_trades"] == 0
    import math

    assert math.isnan(metrics["expectancy"])


def test_profit_factor_infinite_with_no_losses() -> None:
    trades = pd.DataFrame({"pnl": [10.0, 20.0], "return_pct": [0.01, 0.02], "holding_bars": [5, 5]})
    daily_equity = pd.Series([100_000, 100_030], index=pd.date_range("2026-01-05", periods=2))
    metrics = compute_backtest_metrics(trades, daily_equity)
    assert metrics["profit_factor"] == float("inf")
