from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.ict_concepts import (
    ACCUMULATION_WINDOW_MINUTES,
    ICT_FEATURE_COLUMNS,
    add_fvg_features,
    add_ict_features,
    add_po3_features,
)

from .conftest import make_session_minutes


def _bars_for_day(day: str, rows: list[dict]) -> pd.DataFrame:
    ts = make_session_minutes(day)[: len(rows)]
    df = pd.DataFrame(rows)
    df["timestamp"] = ts.to_list()
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def _flat_bar(price: float = 100.0) -> dict:
    return {"open": price, "high": price + 0.1, "low": price - 0.1, "close": price, "volume": 1_000}


# --------------------------------------------------------------------------
# Fair Value Gap
# --------------------------------------------------------------------------


def test_bullish_fvg_detected_and_dist_computed() -> None:
    rows = [_flat_bar() for _ in range(5)]
    # candle i-2: high=100.2
    rows[0] = {"open": 100.0, "high": 100.2, "low": 99.9, "close": 100.0, "volume": 1000}
    rows[1] = _flat_bar(100.3)  # candle i-1, irrelevant to the gap condition
    # candle i: low=101.4 > high[i-2]=100.2 -> bullish FVG
    rows[2] = {"open": 101.5, "high": 101.8, "low": 101.4, "close": 101.5, "volume": 1000}
    bars = _bars_for_day("2026-01-05", rows)

    result = add_fvg_features(bars)

    assert result.loc[2, "bull_fvg_active"] == 1.0
    expected_dist = (result.loc[2, "close"] - result.loc[2, "low"]) / result.loc[2, "close"]
    assert result.loc[2, "bull_fvg_dist"] == pytest.approx(expected_dist)
    assert result.loc[2, "bear_fvg_active"] == 0.0


def test_bearish_fvg_detected() -> None:
    rows = [_flat_bar() for _ in range(5)]
    # candle i-2: low=100.2
    rows[0] = {"open": 100.0, "high": 100.5, "low": 100.2, "close": 100.3, "volume": 1000}
    rows[1] = _flat_bar(99.8)
    # candle i: high=99.7 < low[i-2]=100.2 -> bearish FVG
    rows[2] = {"open": 99.5, "high": 99.7, "low": 99.4, "close": 99.5, "volume": 1000}
    bars = _bars_for_day("2026-01-05", rows)

    result = add_fvg_features(bars)

    assert result.loc[2, "bear_fvg_active"] == 1.0
    assert result.loc[2, "bull_fvg_active"] == 0.0


def test_bullish_fvg_filled_without_inversion() -> None:
    # Gap forms at idx 2: bottom=high[0]=100.2, top=low[2]=101.4.
    rows = [_flat_bar() for _ in range(6)]
    rows[0] = {"open": 100.0, "high": 100.2, "low": 99.9, "close": 100.0, "volume": 1000}
    rows[1] = _flat_bar(100.3)
    rows[2] = {"open": 101.5, "high": 101.8, "low": 101.4, "close": 101.5, "volume": 1000}
    rows[3] = _flat_bar(101.3)
    # Wicks down through the gap bottom (100.2) but closes back above it -> filled, not inverted.
    rows[4] = {"open": 101.0, "high": 101.2, "low": 100.0, "close": 100.9, "volume": 1000}
    bars = _bars_for_day("2026-01-05", rows)

    result = add_fvg_features(bars)

    assert result.loc[4, "bull_fvg_active"] == 0.0
    assert result.loc[4, "bars_since_bull_ifvg"] > 4  # no inversion happened


def test_bullish_fvg_inverted_when_price_closes_through() -> None:
    rows = [_flat_bar() for _ in range(6)]
    rows[0] = {"open": 100.0, "high": 100.2, "low": 99.9, "close": 100.0, "volume": 1000}
    rows[1] = _flat_bar(100.3)
    rows[2] = {"open": 101.5, "high": 101.8, "low": 101.4, "close": 101.5, "volume": 1000}
    rows[3] = _flat_bar(101.3)
    # Closes decisively below the gap bottom (100.2) -> inverted.
    rows[4] = {"open": 100.5, "high": 100.6, "low": 99.5, "close": 99.8, "volume": 1000}
    rows[5] = _flat_bar(99.7)
    bars = _bars_for_day("2026-01-05", rows)

    result = add_fvg_features(bars)

    assert result.loc[4, "bull_fvg_active"] == 0.0
    assert result.loc[4, "bars_since_bull_ifvg"] == 0.0
    assert result.loc[5, "bars_since_bull_ifvg"] == 1.0


def test_fvg_state_resets_each_session() -> None:
    rows_day1 = [_flat_bar() for _ in range(5)]
    rows_day1[0] = {"open": 100.0, "high": 100.2, "low": 99.9, "close": 100.0, "volume": 1000}
    rows_day1[1] = _flat_bar(100.3)
    rows_day1[2] = {"open": 101.5, "high": 101.8, "low": 101.4, "close": 101.5, "volume": 1000}
    day1 = _bars_for_day("2026-01-05", rows_day1)

    rows_day2 = [_flat_bar(200.0) for _ in range(3)]
    day2 = _bars_for_day("2026-01-06", rows_day2)

    bars = pd.concat([day1, day2], ignore_index=True)
    result = add_fvg_features(bars)

    # The bullish gap active at the end of day 1 must not leak into day 2.
    assert result.loc[5, "bull_fvg_active"] == 0.0  # first bar of day 2


# --------------------------------------------------------------------------
# PO3 / AMD
# --------------------------------------------------------------------------


def test_accumulation_range_correct_when_input_is_utc() -> None:
    # Regression test: real cached bars are UTC, not America/New_York like
    # this module's synthetic fixtures — this caught a real bug where the
    # accumulation window silently never matched any bar (session open was
    # computed as midnight-UTC-plus-9:30, not 9:30 ET), leaving
    # dist_to_accum_high/low NaN for 100% of real rows.
    rows = [_flat_bar(100.0) for _ in range(ACCUMULATION_WINDOW_MINUTES + 5)]
    bars = _bars_for_day("2026-01-05", rows)
    bars = bars.assign(timestamp=pd.to_datetime(bars["timestamp"]).dt.tz_convert("UTC"))

    result = add_po3_features(bars)

    assert result["dist_to_accum_high"].isna().sum() == 0
    assert result["dist_to_accum_low"].isna().sum() == 0


def test_accumulation_range_matches_opening_window() -> None:
    rows = []
    for i in range(ACCUMULATION_WINDOW_MINUTES):
        price = 100.0 + (i % 3) * 0.1  # noisy but bounded within the window
        rows.append(
            {"open": price, "high": price + 0.5, "low": price - 0.5, "close": price, "volume": 1000}
        )
    rows.append(_flat_bar(100.0))  # first bar after the window
    bars = _bars_for_day("2026-01-05", rows)

    result = add_po3_features(bars)

    expected_high = bars["high"].iloc[:ACCUMULATION_WINDOW_MINUTES].max()
    expected_low = bars["low"].iloc[:ACCUMULATION_WINDOW_MINUTES].min()
    last_dist_to_high = result["dist_to_accum_high"].iloc[ACCUMULATION_WINDOW_MINUTES]
    assert last_dist_to_high == pytest.approx((100.0 - expected_high) / expected_high)
    last_dist_to_low = result["dist_to_accum_low"].iloc[ACCUMULATION_WINDOW_MINUTES]
    assert last_dist_to_low == pytest.approx((100.0 - expected_low) / expected_low)


def test_sweep_of_accumulation_high_flags_bearish_direction() -> None:
    rows = [_flat_bar(100.0) for _ in range(ACCUMULATION_WINDOW_MINUTES)]  # range: [99.9, 100.1]
    # Pokes above the range high (100.1) then closes back inside -> a sweep.
    rows.append({"open": 100.2, "high": 100.5, "low": 100.1, "close": 100.05, "volume": 1000})
    rows.append(_flat_bar(100.0))
    bars = _bars_for_day("2026-01-05", rows)

    result = add_po3_features(bars)

    sweep_idx = ACCUMULATION_WINDOW_MINUTES
    assert result.loc[sweep_idx, "swept_accum_high_recent"] == 1.0
    assert result.loc[sweep_idx, "sweep_direction"] == -1.0
    # Still "recent" one bar later.
    assert result.loc[sweep_idx + 1, "swept_accum_high_recent"] == 1.0


def test_sweep_of_accumulation_low_flags_bullish_direction() -> None:
    rows = [_flat_bar(100.0) for _ in range(ACCUMULATION_WINDOW_MINUTES)]
    rows.append({"open": 99.9, "high": 99.9, "low": 99.5, "close": 99.95, "volume": 1000})
    bars = _bars_for_day("2026-01-05", rows)

    result = add_po3_features(bars)

    sweep_idx = ACCUMULATION_WINDOW_MINUTES
    assert result.loc[sweep_idx, "swept_accum_low_recent"] == 1.0
    assert result.loc[sweep_idx, "sweep_direction"] == 1.0


def test_no_sweep_flagged_while_range_still_forming() -> None:
    rows = []
    for i in range(ACCUMULATION_WINDOW_MINUTES):
        # Deliberately volatile during the window itself - shouldn't count as a "sweep"
        # since the range isn't finalized yet.
        price = 100.0 + (5 if i == 10 else 0)
        rows.append(
            {"open": price, "high": price + 1, "low": price - 1, "close": price, "volume": 1000}
        )
    bars = _bars_for_day("2026-01-05", rows)

    result = add_po3_features(bars)

    assert (result["swept_accum_high_recent"].iloc[:ACCUMULATION_WINDOW_MINUTES] == 0.0).all()
    assert (result["swept_accum_low_recent"].iloc[:ACCUMULATION_WINDOW_MINUTES] == 0.0).all()


def test_po3_state_resets_each_session() -> None:
    rows_day1 = [_flat_bar(100.0) for _ in range(ACCUMULATION_WINDOW_MINUTES)]
    rows_day1.append({"open": 100.2, "high": 100.5, "low": 100.1, "close": 100.05, "volume": 1000})
    day1 = _bars_for_day("2026-01-05", rows_day1)

    rows_day2 = [_flat_bar(200.0) for _ in range(ACCUMULATION_WINDOW_MINUTES)]
    day2 = _bars_for_day("2026-01-06", rows_day2)

    bars = pd.concat([day1, day2], ignore_index=True)
    result = add_po3_features(bars)

    day2_start = len(day1)
    # Day 2's accumulation range must reflect day 2's own prices (~200), not
    # day 1's leftover state (~100) — a leak would show up as a huge distance
    # (close=200 vs a ~100 range), not this small a gap.
    assert abs(result.loc[day2_start, "dist_to_accum_high"]) < 0.01
    assert result.loc[day2_start, "sweep_direction"] == 0.0  # no sweep recorded yet today


# --------------------------------------------------------------------------
# Lookahead safety
# --------------------------------------------------------------------------


def test_no_lookahead_bias() -> None:
    rows = []
    rng = np.random.default_rng(seed=1)
    price = 100.0
    for _ in range(120):
        price += rng.normal(0, 0.3)
        high = price + abs(rng.normal(0, 0.2))
        low = price - abs(rng.normal(0, 0.2))
        rows.append({"open": price, "high": high, "low": low, "close": price, "volume": 1000})
    bars = _bars_for_day("2026-01-05", rows)

    cutoff = 90
    original = add_ict_features(bars)

    mutated_input = bars.copy()
    mutated_input["volume"] = mutated_input["volume"].astype(float)
    future = mutated_input.index[cutoff:]
    mutated_input.loc[future, "close"] = rng.uniform(1e6, 2e6, size=len(future))
    mutated_input.loc[future, "high"] = mutated_input.loc[future, "close"] * 1.5
    mutated_input.loc[future, "low"] = mutated_input.loc[future, "close"] * 0.5
    mutated_input.loc[future, "open"] = mutated_input.loc[future, "close"]
    mutated_input.loc[future, "volume"] = rng.uniform(1e9, 2e9, size=len(future))

    mutated = add_ict_features(mutated_input)

    for col in ICT_FEATURE_COLUMNS:
        pd.testing.assert_series_equal(
            original[col].iloc[:cutoff], mutated[col].iloc[:cutoff], check_names=False
        )
