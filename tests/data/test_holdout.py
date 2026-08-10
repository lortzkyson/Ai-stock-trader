from __future__ import annotations

import pandas as pd

from data.holdout import exclude_holdout

from .conftest import make_clean_bars


def test_exclude_holdout_drops_rows_inside_the_pinned_window() -> None:
    bars = make_clean_bars(["2026-06-15", "2026-08-03"])

    result = exclude_holdout(bars)

    ts = pd.to_datetime(result["timestamp"])
    assert (ts.dt.date == pd.Timestamp("2026-06-15").date()).sum() == 0
    assert (ts.dt.date == pd.Timestamp("2026-08-03").date()).sum() > 0


def test_exclude_holdout_false_is_a_noop() -> None:
    bars = make_clean_bars(["2026-06-15"])

    result = exclude_holdout(bars, exclude=False)

    pd.testing.assert_frame_equal(result, bars)
