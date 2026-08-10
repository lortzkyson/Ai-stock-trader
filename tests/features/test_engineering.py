from __future__ import annotations

import pandas as pd

from features.engineering import FEATURE_COLUMNS, add_features, assert_no_lookahead

from .conftest import make_clean_bars


def test_add_features_produces_expected_columns() -> None:
    bars = make_clean_bars(["2026-01-05", "2026-01-06"])

    result = add_features(bars)

    for col in FEATURE_COLUMNS:
        assert col in result.columns


def test_no_lookahead_bias() -> None:
    bars = make_clean_bars(["2026-01-05", "2026-01-06", "2026-01-07"])
    assert_no_lookahead(bars)  # raises AssertionError if any feature leaks the future


def test_vwap_dev_is_zero_at_first_bar_of_each_session() -> None:
    bars = make_clean_bars(["2026-01-05", "2026-01-06"])

    result = add_features(bars)
    ts = pd.to_datetime(result["timestamp"])
    first_of_day = ts.dt.date != ts.dt.date.shift(1)

    assert (result.loc[first_of_day, "vwap_dev"].abs() < 1e-9).all()
