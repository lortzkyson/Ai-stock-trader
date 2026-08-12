#!/usr/bin/env python3
"""Stage 1 of the survivorship-bias fix: sweep the full ticker space at
spread-out historical dates to learn which symbols ever traded.

See src/data/ticker_discovery.py for why this is necessary and why Alpaca's
corporate-actions feed is not a substitute.
"""
from __future__ import annotations

import os
import socket
import sys
from datetime import date
from pathlib import Path

from alpaca.data.historical import StockHistoricalDataClient
from dotenv import load_dotenv

# A sweep takes ~an hour of continuous HTTP requests, so it will meet a dropped
# connection — a laptop sleeping mid-run is enough. The Alpaca SDK sets no read
# timeout, so a dead socket blocks forever: the process stays alive, burns no
# CPU, and never advances. This makes it raise instead, so the retry logic in
# data.ticker_discovery can actually do its job.
socket.setdefaulttimeout(60)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.ticker_discovery import discover_master_universe, union_of  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
MASTER_PATH = REPO_ROOT / "data" / "cache" / "master_ticker_universe.txt"

# Semi-annual sweeps: anything listed for more than ~6 months is caught.
PROBE_DATES = [
    date(y, m, d)
    for y in range(2016, 2027)
    for (m, d) in [(6, 15), (12, 15)]
    if date(y, m, d) < date(2026, 8, 1)
]


def main() -> int:
    load_dotenv()
    client = StockHistoricalDataClient(
        os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]
    )
    print(f"Sweeping full ticker space at {len(PROBE_DATES)} dates...")
    by_date = discover_master_universe(client, PROBE_DATES, progress=True)
    master = union_of(by_date)

    MASTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    MASTER_PATH.write_text("\n".join(master))
    print(f"\nMaster universe: {len(master):,} symbols ever traded -> {MASTER_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
