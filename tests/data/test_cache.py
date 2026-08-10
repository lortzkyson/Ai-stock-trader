from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from data.cache import read_cached, write_cache

from .conftest import make_clean_bars


def test_write_then_read_cache_round_trips(tmp_path: Path) -> None:
    bars = make_clean_bars(["2026-01-05"])
    start, end = date(2026, 1, 5), date(2026, 1, 5)

    write_cache(bars, "TEST", "1Min", start, end, cache_dir=tmp_path)
    loaded = read_cached("TEST", "1Min", start, end, cache_dir=tmp_path)

    assert loaded is not None
    pd.testing.assert_frame_equal(loaded, bars)


def test_read_cached_returns_none_when_missing(tmp_path: Path) -> None:
    result = read_cached("NOPE", "1Min", date(2026, 1, 5), date(2026, 1, 5), cache_dir=tmp_path)
    assert result is None
