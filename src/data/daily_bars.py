"""Daily-bar data layer for cross-sectional strategies.

Separate from the minute-bar path in `alpaca_client.py` / `quality.py` for a
real reason, not duplication: those modules do regular-session filtering and
halt-like-day detection, both of which are meaningless for daily bars (a
daily bar *is* the session). The quality checks that do still apply —
duplicates, impossible values, extreme jumps — are re-implemented here
against a multi-symbol panel rather than a single symbol's series.

Data is fetched in batches (Alpaca accepts ~500 symbols per request) and
cached as one combined parquet panel rather than per-symbol files, because
cross-sectional strategies always want the whole panel at once.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.models.bars import BarSet
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PANEL_CACHE_DIR = REPO_ROOT / "data" / "cache" / "daily_panels"

BATCH_SIZE = 500
PANEL_COLUMNS = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]


def _to_datetime(value: date | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.combine(value, time.min, tzinfo=timezone.utc)


def panel_cache_path(
    symbols: list[str], start: date, end: date, cache_dir: Path = DEFAULT_PANEL_CACHE_DIR
) -> Path:
    """Cache key includes a hash of the symbol list — a different universe is a
    different panel, even over the same date range."""
    digest = hashlib.sha256(",".join(sorted(symbols)).encode()).hexdigest()[:12]
    name = f"daily_{start.isoformat()}_{end.isoformat()}_{len(symbols)}sym_{digest}.parquet"
    return cache_dir / name


def fetch_daily_bars(
    client: StockHistoricalDataClient,
    symbols: list[str],
    start: date,
    end: date,
    batch_size: int = BATCH_SIZE,
    progress: bool = False,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        request = StockBarsRequest(
            symbol_or_symbols=batch,
            timeframe=TimeFrame(1, TimeFrameUnit.Day),
            start=_to_datetime(start),
            end=_to_datetime(end),
            feed=DataFeed.SIP,
            adjustment=Adjustment.ALL,
        )
        bars = client.get_stock_bars(request)
        if not isinstance(bars, BarSet):
            raise TypeError(f"expected BarSet from get_stock_bars, got {type(bars)}")
        df = bars.df
        if len(df):
            frames.append(df.reset_index()[PANEL_COLUMNS])
        if progress:
            print(f"  fetched batch {i // batch_size + 1}: {len(batch)} symbols")

    if not frames:
        return pd.DataFrame(columns=PANEL_COLUMNS)
    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def load_or_fetch_daily_panel(
    client: StockHistoricalDataClient,
    symbols: list[str],
    start: date,
    end: date,
    cache_dir: Path = DEFAULT_PANEL_CACHE_DIR,
    progress: bool = False,
) -> pd.DataFrame:
    path = panel_cache_path(symbols, start, end, cache_dir)
    if path.exists():
        return pd.read_parquet(path)

    panel = fetch_daily_bars(client, symbols, start, end, progress=progress)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(path, index=False)
    return panel


@dataclass
class DailyQualityReport:
    n_rows: int
    n_symbols: int
    duplicate_rows: int
    impossible_value_rows: int
    extreme_jump_rows: int

    @property
    def is_clean(self) -> bool:
        return self.duplicate_rows == 0 and self.impossible_value_rows == 0


def check_daily_quality(
    panel: pd.DataFrame, max_single_day_return: float = 1.0
) -> DailyQualityReport:
    duplicates = panel.duplicated(subset=["symbol", "timestamp"])
    non_positive = (panel[["open", "high", "low", "close"]] <= 0).any(axis=1)
    inverted = panel["high"] < panel["low"]
    negative_volume = panel["volume"] < 0
    impossible = non_positive | inverted | negative_volume

    returns = panel.groupby("symbol")["close"].pct_change().abs()
    extreme = returns > max_single_day_return

    return DailyQualityReport(
        n_rows=len(panel),
        n_symbols=panel["symbol"].nunique(),
        duplicate_rows=int(duplicates.sum()),
        impossible_value_rows=int(impossible.sum()),
        extreme_jump_rows=int(extreme.sum()),
    )


def clean_daily_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates and impossible values.

    Extreme single-day moves are reported but NOT dropped: on daily bars a
    >100% move is usually a real event (biotech readout, buyout announcement,
    meme squeeze), and silently removing them would bias a momentum strategy
    precisely where the signal lives.
    """
    duplicates = panel.duplicated(subset=["symbol", "timestamp"])
    non_positive = (panel[["open", "high", "low", "close"]] <= 0).any(axis=1)
    inverted = panel["high"] < panel["low"]
    negative_volume = panel["volume"] < 0
    keep = ~(duplicates | non_positive | inverted | negative_volume)
    return panel.loc[keep].reset_index(drop=True)
