"""Assemble the modeling dataset: cached bars -> clean -> label -> features -> combined table."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from data.cache import DEFAULT_CACHE_DIR, read_cached
from data.holdout import exclude_holdout
from data.quality import check_quality, clean_bars
from features.engineering import FEATURE_COLUMNS, add_features
from features.labeling import TripleBarrierConfig, label_triple_barrier


def build_symbol_dataset(
    symbol: str,
    start: date,
    end: date,
    barrier_config: TripleBarrierConfig | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    raw = read_cached(symbol, "1Min", start, end, cache_dir)
    if raw is None:
        raise FileNotFoundError(
            f"no cached bars for {symbol} {start}-{end}; "
            "run scripts/fetch_training_data.py first"
        )

    report = check_quality(raw, symbol)
    bars = clean_bars(raw, report)
    bars = exclude_holdout(bars)
    labeled = label_triple_barrier(bars, barrier_config)
    featured = add_features(labeled)
    featured["symbol"] = symbol
    return featured


def build_dataset(
    symbols: list[str],
    start: date,
    end: date,
    barrier_config: TripleBarrierConfig | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    frames = [
        build_symbol_dataset(s, start, end, barrier_config, cache_dir) for s in symbols
    ]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["label", *FEATURE_COLUMNS]).copy()
    combined["date"] = pd.to_datetime(combined["timestamp"]).dt.date
    combined["target"] = (combined["label"] == 1).astype(int)
    return combined.sort_values("timestamp").reset_index(drop=True)
