from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.momentum import (
    MomentumConfig,
    build_price_matrix,
    build_target_portfolios,
    compute_momentum_scores,
    month_end_rebalance_dates,
    select_positions,
)


def make_panel(prices_by_symbol: dict[str, list[float]], start: str = "2026-01-01",
               volume: float = 1_000_000) -> pd.DataFrame:
    rows = []
    n = len(next(iter(prices_by_symbol.values())))
    dates = pd.bdate_range(start, periods=n)
    for symbol, prices in prices_by_symbol.items():
        for ts, price in zip(dates, prices):
            rows.append({
                "symbol": symbol, "timestamp": ts, "open": price, "high": price * 1.01,
                "low": price * 0.99, "close": price, "volume": volume,
            })
    return pd.DataFrame(rows)


def test_momentum_score_measures_return_over_formation_window_skipping_recent() -> None:
    # 10 bars. With formation=5, skip=2, the score at index 7 should be
    # price[5]/price[2] - 1 — i.e. it ignores the most recent 2 bars entirely.
    prices = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    panel = make_panel({"AAA": prices})
    matrix = build_price_matrix(panel)

    scores = compute_momentum_scores(matrix, MomentumConfig(formation_days=5, skip_days=2))

    expected = prices[5] / prices[2] - 1
    assert scores["AAA"].iloc[7] == pytest.approx(expected)


def test_momentum_score_is_nan_before_enough_history() -> None:
    panel = make_panel({"AAA": [10.0] * 10})
    matrix = build_price_matrix(panel)

    scores = compute_momentum_scores(matrix, MomentumConfig(formation_days=5, skip_days=2))

    assert scores["AAA"].iloc[:5].isna().all()


def test_recent_move_does_not_affect_score_confirming_skip_window() -> None:
    """The skip window exists because short-horizon returns mean-revert. A huge
    move in the skipped period must not change the score."""
    base = [10, 11, 12, 13, 14, 15, 16, 17]
    spiked = base[:-2] + [100.0, 200.0]  # enormous move in the last 2 (skipped) bars
    config = MomentumConfig(formation_days=5, skip_days=2)

    s_base = compute_momentum_scores(build_price_matrix(make_panel({"AAA": base})), config)
    s_spiked = compute_momentum_scores(build_price_matrix(make_panel({"AAA": spiked})), config)

    assert s_base["AAA"].iloc[7] == pytest.approx(s_spiked["AAA"].iloc[7])


def test_select_positions_ranks_by_score_and_respects_n() -> None:
    dates = pd.bdate_range("2026-01-01", periods=1)
    scores = pd.DataFrame({"AAA": [0.5], "BBB": [0.9], "CCC": [0.1]}, index=dates)
    prices = pd.DataFrame({"AAA": [50.0], "BBB": [60.0], "CCC": [70.0]}, index=dates)
    dv = pd.DataFrame({"AAA": [1e9], "BBB": [1e9], "CCC": [1e9]}, index=dates)

    selected = select_positions(scores, prices, dv, dates[0], MomentumConfig(n_positions=2))

    assert selected == ["BBB", "AAA"]


def test_select_positions_excludes_penny_and_illiquid_names_before_ranking() -> None:
    dates = pd.bdate_range("2026-01-01", periods=1)
    # PENNY has the best score but fails the price floor; THIN fails liquidity.
    scores = pd.DataFrame({"GOOD": [0.2], "PENNY": [9.9], "THIN": [5.0]}, index=dates)
    prices = pd.DataFrame({"GOOD": [50.0], "PENNY": [1.0], "THIN": [80.0]}, index=dates)
    dv = pd.DataFrame({"GOOD": [1e9], "PENNY": [1e9], "THIN": [1_000.0]}, index=dates)

    selected = select_positions(scores, prices, dv, dates[0], MomentumConfig(n_positions=5))

    assert selected == ["GOOD"]


def test_select_positions_skips_symbols_with_nan_score() -> None:
    dates = pd.bdate_range("2026-01-01", periods=1)
    scores = pd.DataFrame({"AAA": [np.nan], "BBB": [0.3]}, index=dates)
    prices = pd.DataFrame({"AAA": [50.0], "BBB": [60.0]}, index=dates)
    dv = pd.DataFrame({"AAA": [1e9], "BBB": [1e9]}, index=dates)

    selected = select_positions(scores, prices, dv, dates[0], MomentumConfig(n_positions=5))

    assert selected == ["BBB"]


def test_month_end_rebalance_dates_picks_last_trading_day_per_month() -> None:
    dates = pd.bdate_range("2026-01-01", "2026-03-31")
    prices = pd.DataFrame({"AAA": range(len(dates))}, index=dates)

    rebalances = month_end_rebalance_dates(prices)

    assert len(rebalances) == 3
    assert rebalances[0] == pd.Timestamp("2026-01-30")  # last business day of Jan
    assert all(r in dates for r in rebalances)


def test_build_target_portfolios_produces_monthly_targets() -> None:
    n = 300
    rng = np.random.default_rng(0)
    panel = make_panel({
        f"S{i}": list(100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n)))
        for i in range(10)
    })

    config = MomentumConfig(formation_days=60, skip_days=10, n_positions=3,
                            min_dollar_volume=0.0, dollar_volume_window=5)
    targets, prices = build_target_portfolios(panel, config)

    assert len(targets) > 0
    for _, symbols in targets.items():
        assert len(symbols) <= 3
    assert prices.shape[1] == 10


def test_no_lookahead_future_prices_cannot_change_past_scores() -> None:
    n = 100
    rng = np.random.default_rng(3)
    prices = list(100 * np.cumprod(1 + rng.normal(0, 0.01, n)))
    config = MomentumConfig(formation_days=30, skip_days=5)

    original = compute_momentum_scores(build_price_matrix(make_panel({"AAA": prices})), config)

    mutated_prices = prices[:70] + [p * 50 for p in prices[70:]]
    mutated = compute_momentum_scores(
        build_price_matrix(make_panel({"AAA": mutated_prices})), config
    )

    pd.testing.assert_series_equal(
        original["AAA"].iloc[:70], mutated["AAA"].iloc[:70], check_names=False
    )
