from __future__ import annotations

import pandas as pd

from data.quality import (
    check_quality,
    clean_bars,
    find_duplicate_timestamps,
    find_extreme_jumps,
    find_halt_like_days,
    find_impossible_values,
)

from .conftest import make_clean_bars


def test_clean_synthetic_data_has_no_flags() -> None:
    bars = make_clean_bars(["2026-01-05"])
    report = check_quality(bars, "TEST")
    assert report.is_clean
    assert report.duplicate_timestamp_count == 0
    assert report.impossible_value_count == 0
    assert report.halt_like_days == []


def test_find_duplicate_timestamps() -> None:
    bars = make_clean_bars(["2026-01-05"])
    dup_row = bars.iloc[[0]]
    bars_with_dup = pd.concat([bars, dup_row], ignore_index=True)

    mask = find_duplicate_timestamps(bars_with_dup)

    assert mask.sum() == 1
    assert mask.iloc[-1]


def test_find_impossible_values_flags_non_positive_and_inverted_range() -> None:
    bars = make_clean_bars(["2026-01-05"]).copy()
    bars.loc[0, "close"] = -5.0
    bars.loc[1, "high"] = bars.loc[1, "low"] - 1.0

    mask = find_impossible_values(bars)

    assert mask.iloc[0]
    assert mask.iloc[1]
    assert mask.sum() == 2


def test_find_extreme_jumps() -> None:
    bars = make_clean_bars(["2026-01-05"]).copy()
    bars.loc[5, "close"] = bars.loc[4, "close"] * 3

    mask = find_extreme_jumps(bars, max_single_bar_return=0.5)

    # The spike shows up as two suspicious bars: the jump up into row 5, and
    # the jump back down into row 6 relative to the now-inflated row 5.
    assert mask.iloc[5]
    assert mask.iloc[6]
    assert mask.sum() == 2


def test_find_halt_like_days_flags_large_gap_but_not_clean_days() -> None:
    bars = make_clean_bars(["2026-01-05", "2026-01-06"])
    day2 = pd.Timestamp("2026-01-06", tz="America/New_York")
    ts = pd.to_datetime(bars["timestamp"])
    drop_mask = (ts.dt.date == day2.date()) & (ts.dt.hour == 11)
    bars_with_halt = bars.loc[~drop_mask].reset_index(drop=True)

    flagged = find_halt_like_days(bars_with_halt, halt_gap_minutes=5)

    flagged_dates = {d.date() for d in flagged}
    assert day2.date() in flagged_dates
    assert pd.Timestamp("2026-01-05").date() not in flagged_dates


def test_clean_bars_drops_dupes_impossible_values_and_halt_days() -> None:
    bars = make_clean_bars(["2026-01-05", "2026-01-06"])
    bars = pd.concat([bars, bars.iloc[[0]]], ignore_index=True)
    bars.loc[10, "close"] = 0.0

    day2 = pd.Timestamp("2026-01-06", tz="America/New_York")
    ts = pd.to_datetime(bars["timestamp"])
    drop_mask = (ts.dt.date == day2.date()) & (ts.dt.hour == 11)
    bars = bars.loc[~drop_mask].reset_index(drop=True)

    report = check_quality(bars, "TEST")
    assert not report.is_clean

    cleaned = clean_bars(bars, report)
    cleaned_ts = pd.to_datetime(cleaned["timestamp"])

    assert (cleaned_ts.dt.date == day2.date()).sum() == 0
    assert (cleaned["close"] <= 0).sum() == 0
    assert not cleaned_ts.duplicated().any()
