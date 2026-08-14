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


def fetch_daily_bars_verified(
    client: StockHistoricalDataClient,
    symbols: list[str],
    start: date,
    end: date,
    batch_size: int = BATCH_SIZE,
    progress: bool = False,
) -> pd.DataFrame:
    """Fetch bars, then re-request anything that came back empty.

    Alpaca silently collapses tickers that resolve to the same underlying
    security. Requesting FRC and FRCB together (First Republic's NYSE listing
    and its post-failure OTC ticker) returns only FRCB — FRC vanishes with no
    error, no warning, and no missing-symbol field. Both return data fine when
    requested apart.

    That's a serious hazard for survivorship-bias correction specifically,
    because delisted companies frequently pick up an aliased OTC ticker: the
    very names being recovered are the ones most likely to be silently dropped,
    and the failure looks identical to "this symbol legitimately has no data".

    The second pass fixes it because only one side of a collision goes missing
    — the winner already returned data, so it isn't in the retry set and can't
    collide again. Symbols still empty after the retry are genuinely absent.
    """
    panel = fetch_daily_bars(client, symbols, start, end, batch_size, progress)
    returned = set(panel["symbol"].unique()) if len(panel) else set()
    missing = [s for s in symbols if s not in returned]
    if not missing:
        return panel

    if progress:
        print(f"  verifying {len(missing):,} symbols that returned no data...")
    retry = fetch_daily_bars(client, missing, start, end, batch_size, progress)
    if not len(retry):
        return panel

    recovered = retry["symbol"].nunique()
    if progress and recovered:
        print(f"  recovered {recovered:,} symbols on retry (alias collisions)")
    combined = pd.concat([panel, retry], ignore_index=True)
    return combined.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


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

    # Verified variant: without it, aliased tickers recovered by the
    # survivorship fix would be silently dropped again here.
    panel = fetch_daily_bars_verified(client, symbols, start, end, progress=progress)
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


def split_discontinuous_series(
    panel: pd.DataFrame, max_daily_return: float = 3.0
) -> pd.DataFrame:
    """Drop history preceding a price discontinuity that implies a new security.

    When a company goes through bankruptcy and reorganizes, the ticker often
    survives but the equity does not: old shares are cancelled at pennies and
    new shares are issued under the same symbol. Alpaca's adjustment pipeline
    doesn't correct for this — it isn't a split — so the raw series contains a
    fabricated overnight gain. Measured on the survivorship-corrected universe:
    GPOR +52,648% (\\$0.14 -> \\$72.95), LINE +44,787%, OAS +19,893%, and 412
    days panel-wide exceeding +100%.

    This matters far more than it sounds. These artifacts cluster in beaten-down
    names, which is exactly what a momentum short leg selects and what a
    survivorship-bias correction pulls back in — so the correction that made the
    backtest honest also imported the contamination. Left alone it produced a
    -84% single-day portfolio loss.

    Everything before the discontinuity is nulled rather than the symbol being
    dropped: the post-reorganization company is a legitimate tradable security,
    it just has no valid price history joining it to its predecessor. Momentum
    scores spanning the break become NaN and the name is simply ineligible until
    clean history accumulates.

    3.0 (a 300% single-day move) is deliberately loose. Genuine squeezes reach
    100-200%; nothing real reaches 300% in a session, so this only catches
    artifacts.
    """
    if panel.empty:
        return panel

    out = panel.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    returns = out.groupby("symbol")["close"].pct_change(fill_method=None)
    breaks = returns.abs() > max_daily_return
    if not breaks.any():
        return out

    # For each affected symbol keep only rows at/after its LAST discontinuity.
    break_rows = out.loc[breaks, ["symbol"]].copy()
    break_rows["row"] = break_rows.index
    last_break_idx = break_rows.groupby("symbol")["row"].max()
    drop_mask = pd.Series(False, index=out.index)
    for symbol, break_idx in last_break_idx.items():
        symbol_rows = out.index[out["symbol"] == symbol]
        drop_mask.loc[symbol_rows[symbol_rows < break_idx]] = True

    return out.loc[~drop_mask].reset_index(drop=True)


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
    cleaned = panel.loc[keep].reset_index(drop=True)
    return split_discontinuous_series(cleaned)
