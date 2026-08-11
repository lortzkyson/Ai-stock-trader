from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.portfolio_metrics import (
    benchmark_series,
    compare_to_benchmark,
    compute_portfolio_metrics,
    equal_weight_buy_and_hold,
    volatility_matched_benchmark,
)


def _series(values: list[float], start: str = "2026-01-01") -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)))


def test_compute_portfolio_metrics_basic() -> None:
    equity = _series([100.0, 101.0, 102.0, 101.0, 103.0])

    m = compute_portfolio_metrics(equity)

    assert m["total_return"] == pytest.approx(0.03)
    assert m["max_drawdown"] < 0
    assert m["n_days"] == 5


def test_max_drawdown_measures_peak_to_trough() -> None:
    equity = _series([100.0, 120.0, 90.0, 95.0])
    m = compute_portfolio_metrics(equity)
    assert m["max_drawdown"] == pytest.approx(90 / 120 - 1)


def test_metrics_handle_too_short_series() -> None:
    m = compute_portfolio_metrics(_series([100.0]))
    assert np.isnan(m["total_return"])


def test_benchmark_series_normalizes_to_starting_equity() -> None:
    prices = pd.DataFrame(
        {"SPY": [400.0, 420.0, 440.0]}, index=pd.bdate_range("2026-01-01", periods=3)
    )
    bench = benchmark_series(prices, "SPY", starting_equity=10_000)
    assert bench.iloc[0] == pytest.approx(10_000)
    assert bench.iloc[-1] == pytest.approx(11_000)


def test_equal_weight_buy_and_hold_splits_capital_evenly() -> None:
    prices = pd.DataFrame(
        {"AAA": [100.0, 200.0], "BBB": [50.0, 50.0]},
        index=pd.bdate_range("2026-01-01", periods=2),
    )
    equity = equal_weight_buy_and_hold(prices, ["AAA", "BBB"], starting_equity=10_000)
    # $5k each; AAA doubles, BBB flat -> $10k + $5k = $15k
    assert equity.iloc[0] == pytest.approx(10_000)
    assert equity.iloc[-1] == pytest.approx(15_000)


def test_compare_to_benchmark_detects_no_real_difference() -> None:
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0005, 0.01, 500)
    equity = _series(list(10_000 * np.cumprod(1 + returns)))

    result = compare_to_benchmark(equity, equity.copy())

    assert result["excess_total_return"] == pytest.approx(0.0, abs=1e-9)
    assert not result["is_significant"]


def test_compare_to_benchmark_detects_a_real_difference() -> None:
    rng = np.random.default_rng(1)
    bench_returns = rng.normal(0.0002, 0.01, 750)
    # Consistent daily outperformance -> should be detectable.
    strat_returns = bench_returns + 0.001

    bench = _series(list(10_000 * np.cumprod(1 + bench_returns)))
    strat = _series(list(10_000 * np.cumprod(1 + strat_returns)))

    result = compare_to_benchmark(strat, bench)

    assert result["excess_total_return"] > 0
    assert result["is_significant"]


def test_volatility_matched_benchmark_recovers_the_leverage_factor() -> None:
    rng = np.random.default_rng(5)
    bench_returns = rng.normal(0.0003, 0.01, 1000)
    # Strategy is exactly 2x the benchmark's returns -> 2x its volatility.
    strat_returns = bench_returns * 2

    bench = _series(list(10_000 * np.cumprod(1 + bench_returns)))
    strat = _series(list(10_000 * np.cumprod(1 + strat_returns)))

    levered, leverage = volatility_matched_benchmark(strat, bench)

    assert leverage == pytest.approx(2.0, rel=0.02)
    assert len(levered) > 0


def test_pure_leverage_shows_no_edge_against_vol_matched_benchmark() -> None:
    """The whole point of this benchmark: a strategy that is *only* levered
    beta must show no significant excess once volatility is matched.

    Drift is set well above the noise here on purpose. At low drift, leverage
    can actually *reduce* total return via volatility drag (variance grows with
    the square of leverage while expected return grows linearly) — true, but it
    muddies what this test is checking.
    """
    rng = np.random.default_rng(7)
    bench_returns = rng.normal(0.0008, 0.01, 1500)
    strat_returns = bench_returns * 2.5  # leverage alone, zero skill

    bench = _series(list(10_000 * np.cumprod(1 + bench_returns)))
    strat = _series(list(10_000 * np.cumprod(1 + strat_returns)))

    # Against the raw benchmark the levered strategy looks far better...
    raw = compare_to_benchmark(strat, bench)
    assert raw["excess_total_return"] > 0

    # ...but against a vol-matched benchmark the apparent edge disappears.
    levered, _ = volatility_matched_benchmark(strat, bench)
    matched = compare_to_benchmark(strat, levered)
    assert not matched["is_significant"]


def test_volatility_matched_benchmark_handles_short_series() -> None:
    _, leverage = volatility_matched_benchmark(_series([1.0, 2.0]), _series([1.0, 2.0]))
    assert np.isnan(leverage)


def test_compare_to_benchmark_handles_empty_overlap() -> None:
    a = _series([1.0, 2.0], start="2026-01-01")
    b = _series([1.0, 2.0], start="2027-01-01")
    result = compare_to_benchmark(a, b)
    assert not result["is_significant"]
