"""Ticker universe and liquidity filtering.

Survivorship bias (known limitation, see docs/pre-mortem.md guard #2 and
README "Known limitations"): data/universe.csv is a manually curated list of
today's liquid large-cap names, not a point-in-time index membership feed.
It excludes tickers that were delisted, acquired, or went bankrupt during any
historical backtest period, so backtests run against this universe will
overstate performance relative to a survivorship-bias-free universe. Getting
point-in-time constituent data typically requires a paid reference-data
subscription; this is deferred rather than solved here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
UNIVERSE_FILE = REPO_ROOT / "data" / "universe.csv"


def load_seed_universe(path: Path = UNIVERSE_FILE) -> list[str]:
    df = pd.read_csv(path)
    return df["symbol"].tolist()


@dataclass(frozen=True)
class LiquidityFilterConfig:
    min_avg_dollar_volume: float = 5_000_000.0
    min_price: float = 5.0
    max_price: float = 1_000.0


def filter_by_liquidity(
    daily_bars: pd.DataFrame,
    config: LiquidityFilterConfig | None = None,
) -> list[str]:
    """Return symbols meeting minimum dollar-volume and price-range criteria.

    `daily_bars` must already be limited to the lookback window the caller
    wants evaluated (e.g. the most recent 20 trading days) and must have
    columns: symbol, close, volume.
    """
    config = config or LiquidityFilterConfig()
    required = {"symbol", "close", "volume"}
    missing = required - set(daily_bars.columns)
    if missing:
        raise ValueError(f"daily_bars missing required columns: {sorted(missing)}")

    working = daily_bars.assign(dollar_volume=daily_bars["close"] * daily_bars["volume"])
    stats = working.groupby("symbol").agg(
        avg_dollar_volume=("dollar_volume", "mean"),
        avg_price=("close", "mean"),
    )

    keep = stats[
        (stats["avg_dollar_volume"] >= config.min_avg_dollar_volume)
        & (stats["avg_price"] >= config.min_price)
        & (stats["avg_price"] <= config.max_price)
    ]
    return keep.index.tolist()
