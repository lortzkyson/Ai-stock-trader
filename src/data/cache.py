"""Local parquet cache so backtests are reproducible and don't re-hit the API."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "cache"


def cache_path(
    symbol: str,
    timeframe: str,
    start: date,
    end: date,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Path:
    return cache_dir / timeframe / f"{symbol}_{start.isoformat()}_{end.isoformat()}.parquet"


def read_cached(
    symbol: str,
    timeframe: str,
    start: date,
    end: date,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> pd.DataFrame | None:
    path = cache_path(symbol, timeframe, start, end, cache_dir)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def write_cache(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    start: date,
    end: date,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Path:
    path = cache_path(symbol, timeframe, start, end, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path
