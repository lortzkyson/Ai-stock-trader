"""Discover which tickers actually existed on historical dates.

This is the fix for survivorship bias. Alpaca's asset list only contains
*currently* listed symbols, so any company that delisted, went bankrupt, or was
acquired is invisible — but Alpaca will happily serve historical bars for those
tickers if you know to ask. The missing piece is the list, and it can be
recovered by brute force: ask for bars on a historical date across the whole
plausible ticker space and keep whatever comes back.

Why not the obvious alternatives:
- **Alpaca's corporate-actions feed** captures mergers but almost no
  bankruptcies (measured: 2 of 16 known failures — SVB, First Republic,
  Signature, Bed Bath & Beyond, WeWork and Rite Aid are all absent). Using it
  alone would *worsen* the bias, since it adds companies acquired at a premium
  while still hiding the ones that went to zero.
- **SEC EDGAR Form 25** filings are authoritative and free, but require parsing
  tens of thousands of filings and mapping CIK to ticker for dead companies.

Two stages, because a full-space probe is ~950 requests:
1. `discover_master_universe` sweeps the entire 1-4 letter space at a handful of
   spread-out dates to learn which tickers *ever* traded (~10-15k of 475k).
2. `probe_existing_symbols` then re-probes just that master list at monthly
   granularity, which is ~30 requests instead of 950 — cheap enough to build a
   true point-in-time universe for every rebalance date.

Results are cached per probe date so a long sweep survives interruption.
"""

from __future__ import annotations

import itertools
import json
import string
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient

from data.daily_bars import fetch_daily_bars

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DISCOVERY_CACHE = REPO_ROOT / "data" / "cache" / "ticker_discovery"

PROBE_WINDOW_DAYS = 7
PROBE_BATCH_SIZE = 500
MAX_BATCH_ATTEMPTS = 4
RETRY_BACKOFF_SECONDS = 3.0


def _fetch_batch_with_retry(
    client: StockHistoricalDataClient,
    batch: list[str],
    start: date,
    end: date,
) -> pd.DataFrame:
    """Fetch one batch, retrying transient network failures.

    A sweep runs for the better part of an hour, so it *will* meet a dropped
    connection — a laptop sleeping mid-run is enough. Without this, the
    underlying HTTP read blocks indefinitely and the whole sweep silently
    wedges: the process stays alive, burns no CPU, and never advances. That
    happened. Callers should also set a socket-level default timeout so a dead
    connection raises instead of hanging forever (see scripts/discover_universe.py).
    """
    last_error: Exception | None = None
    for attempt in range(1, MAX_BATCH_ATTEMPTS + 1):
        try:
            return fetch_daily_bars(client, batch, start, end, batch_size=PROBE_BATCH_SIZE)
        except Exception as exc:  # network/API errors vary by layer
            last_error = exc
            if attempt < MAX_BATCH_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise RuntimeError(
        f"batch of {len(batch)} symbols failed after {MAX_BATCH_ATTEMPTS} attempts"
    ) from last_error


def all_candidate_tickers(max_len: int = 4) -> list[str]:
    """Every 1..max_len uppercase-letter combination.

    Deliberately excludes 5-letter symbols: on US exchanges those are
    overwhelmingly OTC issues, mutual funds, warrants and preferred shares
    rather than listed common stock, and including them would multiply the
    probe space by ~25x for candidates the liquidity filter would reject anyway.
    """
    out: list[str] = []
    for n in range(1, max_len + 1):
        out.extend("".join(p) for p in itertools.product(string.ascii_uppercase, repeat=n))
    return out


def _cache_path(probe_date: date, cache_dir: Path) -> Path:
    return cache_dir / f"probe_{probe_date.isoformat()}.json"


def probe_existing_symbols(
    client: StockHistoricalDataClient,
    candidates: list[str],
    probe_date: date,
    window_days: int = PROBE_WINDOW_DAYS,
    cache_dir: Path = DEFAULT_DISCOVERY_CACHE,
    use_cache: bool = True,
    progress: bool = False,
) -> set[str]:
    """Symbols with at least one daily bar in the window ending at `probe_date`."""
    path = _cache_path(probe_date, cache_dir)
    if use_cache and path.exists():
        return set(json.loads(path.read_text()))

    start = probe_date - timedelta(days=window_days)
    found: set[str] = set()
    for i in range(0, len(candidates), PROBE_BATCH_SIZE):
        batch = candidates[i : i + PROBE_BATCH_SIZE]
        panel = _fetch_batch_with_retry(client, batch, start, probe_date)
        if len(panel):
            found.update(panel["symbol"].unique())
        if progress and (i // PROBE_BATCH_SIZE) % 100 == 0:
            print(f"    probed {i + len(batch):,}/{len(candidates):,} — {len(found):,} found")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(found)))
    return found


def discover_master_universe(
    client: StockHistoricalDataClient,
    probe_dates: list[date],
    max_len: int = 4,
    cache_dir: Path = DEFAULT_DISCOVERY_CACHE,
    progress: bool = False,
) -> dict[date, set[str]]:
    """Full-space sweep at each date. Returns {probe_date: symbols alive then}.

    Known gap: a ticker that both listed and delisted strictly between two probe
    dates is missed entirely. Spacing the probes closer narrows this; it can't
    be eliminated without a true point-in-time reference feed.
    """
    candidates = all_candidate_tickers(max_len)
    if progress:
        print(f"  candidate ticker space: {len(candidates):,}")

    by_date: dict[date, set[str]] = {}
    for probe_date in probe_dates:
        found = probe_existing_symbols(
            client, candidates, probe_date, cache_dir=cache_dir, progress=progress
        )
        by_date[probe_date] = found
        if progress:
            print(f"  {probe_date}: {len(found):,} symbols alive")
    return by_date


def union_of(by_date: dict[date, set[str]]) -> list[str]:
    universe: set[str] = set()
    for symbols in by_date.values():
        universe.update(symbols)
    return sorted(universe)
