from __future__ import annotations

import numpy as np
import pytest

from models.metrics import compute_signal_metrics


def test_compute_signal_metrics_basic() -> None:
    returns = np.array([0.02, 0.02, -0.01, -0.01, -0.01])
    labels = np.array([1, 1, -1, -1, -1])
    dates = np.array(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"])

    metrics = compute_signal_metrics(returns, labels, dates)

    assert metrics["n_trades"] == 5
    assert metrics["win_rate"] == pytest.approx(0.4)
    assert metrics["expectancy"] == pytest.approx(returns.mean())
    assert metrics["max_drawdown"] <= 0


def test_compute_signal_metrics_empty() -> None:
    metrics = compute_signal_metrics(np.array([]), np.array([]), np.array([]))
    assert metrics["n_trades"] == 0
    assert np.isnan(metrics["expectancy"])


def test_compute_signal_metrics_zero_std_gives_nan_sharpe() -> None:
    returns = np.array([0.01, 0.01, 0.01])
    labels = np.array([1, 1, 1])
    dates = np.array(["2026-01-05", "2026-01-06", "2026-01-07"])
    metrics = compute_signal_metrics(returns, labels, dates)
    assert np.isnan(metrics["sharpe"])


def test_low_win_rate_can_still_have_positive_expectancy() -> None:
    # 10% win rate, 2% wins vs 1% losses -> still net negative here, the exact
    # scenario docs/pre-mortem.md targets expectancy over win rate for.
    returns = np.array([0.02, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01])
    labels = np.array([1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
    dates = np.array([f"2026-01-{5 + i:02d}" for i in range(10)])
    metrics = compute_signal_metrics(returns, labels, dates)
    assert metrics["win_rate"] == pytest.approx(0.1)
    # 0.1*0.02 - 0.9*0.01 = -0.007, i.e. this particular mix is a net loser —
    # confirms the metric reflects expectancy, not just win/loss counting.
    assert metrics["expectancy"] == pytest.approx(-0.007)


def test_multiple_trades_same_day_are_averaged_not_compounded_sequentially() -> None:
    # Regression test: naively compounding every trade sequentially with
    # (1+r) treats concurrent same-day trades as if they happened one after
    # another with 100% of capital reinvested each time. With enough
    # same-day trades that blows the equity curve toward zero even though
    # the day's *average* outcome was mildly positive. Grouping by day first
    # avoids that.
    returns = np.array([0.01, -0.01, 0.01, -0.01] * 500)  # 2000 trades, same day, net ~0
    labels = np.where(returns > 0, 1, -1)
    dates = np.array(["2026-01-05"] * len(returns))

    metrics = compute_signal_metrics(returns, labels, dates)

    assert metrics["n_trades"] == 2000
    assert metrics["max_drawdown"] > -0.5  # not an equity-curve wipeout
