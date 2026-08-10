from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from data.pipeline import run_pipeline

from .conftest import make_clean_bars


def test_run_pipeline_fetches_cleans_and_caches(tmp_path: Path) -> None:
    fetch_calls: list[str] = []

    def fake_fetch(symbol: str, start: date, end: date) -> pd.DataFrame:
        fetch_calls.append(symbol)
        bars = make_clean_bars(["2026-01-05"])
        bars.loc[0, "close"] = -1.0  # inject an impossible value to check cleaning happens
        return bars

    result = run_pipeline(
        symbols=["AAA", "BBB"],
        start=date(2026, 1, 5),
        end=date(2026, 1, 5),
        fetch_fn=fake_fetch,
        cache_dir=tmp_path,
    )

    assert sorted(fetch_calls) == ["AAA", "BBB"]
    assert set(result.bars) == {"AAA", "BBB"}
    assert (result.bars["AAA"]["close"] <= 0).sum() == 0
    assert result.quality["AAA"].impossible_value_count == 1


def test_run_pipeline_uses_cache_and_skips_refetch(tmp_path: Path) -> None:
    fetch_calls: list[str] = []

    def fake_fetch(symbol: str, start: date, end: date) -> pd.DataFrame:
        fetch_calls.append(symbol)
        return make_clean_bars(["2026-01-05"])

    common_kwargs = dict(
        symbols=["AAA"],
        start=date(2026, 1, 5),
        end=date(2026, 1, 5),
        fetch_fn=fake_fetch,
        cache_dir=tmp_path,
    )

    run_pipeline(**common_kwargs)
    run_pipeline(**common_kwargs)

    assert fetch_calls == ["AAA"]  # second run served from cache, no re-fetch


def test_run_pipeline_use_cache_false_always_refetches(tmp_path: Path) -> None:
    fetch_calls: list[str] = []

    def fake_fetch(symbol: str, start: date, end: date) -> pd.DataFrame:
        fetch_calls.append(symbol)
        return make_clean_bars(["2026-01-05"])

    common_kwargs = dict(
        symbols=["AAA"],
        start=date(2026, 1, 5),
        end=date(2026, 1, 5),
        fetch_fn=fake_fetch,
        cache_dir=tmp_path,
        use_cache=False,
    )

    run_pipeline(**common_kwargs)
    run_pipeline(**common_kwargs)

    assert fetch_calls == ["AAA", "AAA"]
