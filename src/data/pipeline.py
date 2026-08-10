"""End-to-end per-symbol pipeline: cache-or-fetch -> quality-check -> clean.

`fetch_fn` is injected rather than importing AlpacaBarsClient directly so the
pipeline can be tested against a synthetic fetch function with no network
calls (see tests/data/test_pipeline.py) and reused later against a real
client in Phase 5/7.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

import pandas as pd

from data.cache import DEFAULT_CACHE_DIR, read_cached, write_cache
from data.quality import QualityReport, check_quality, clean_bars


class BarsFetcher(Protocol):
    def __call__(self, symbol: str, start: date, end: date) -> pd.DataFrame: ...


@dataclass
class PipelineResult:
    bars: dict[str, pd.DataFrame]
    quality: dict[str, QualityReport]


def run_pipeline(
    symbols: list[str],
    start: date,
    end: date,
    fetch_fn: BarsFetcher,
    timeframe: str = "1Min",
    use_cache: bool = True,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> PipelineResult:
    bars: dict[str, pd.DataFrame] = {}
    quality: dict[str, QualityReport] = {}

    for symbol in symbols:
        raw = read_cached(symbol, timeframe, start, end, cache_dir) if use_cache else None
        if raw is None:
            raw = fetch_fn(symbol, start, end)
            write_cache(raw, symbol, timeframe, start, end, cache_dir)

        report = check_quality(raw, symbol)
        quality[symbol] = report
        bars[symbol] = clean_bars(raw, report)

    return PipelineResult(bars=bars, quality=quality)
