#!/usr/bin/env python3
"""Stage 2: liquidity-screen the survivorship-corrected master universe.

Reduces the ~18.6k symbols that ever traded down to those actually investable,
so the momentum backtest doesn't need a decade of history for thousands of
illiquid names. Screening runs at several points in time and unions the
results, so a name that was liquid in 2018 but delisted in 2021 still qualifies.
"""
from __future__ import annotations

import os
import socket
import sys
from datetime import date
from pathlib import Path

from alpaca.data.historical import StockHistoricalDataClient
from dotenv import load_dotenv

socket.setdefaulttimeout(60)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.universe_screen import ScreenConfig, build_union_universe  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
MASTER = REPO_ROOT / "data" / "cache" / "master_ticker_universe.txt"
OUT = REPO_ROOT / "data" / "cache" / "momentum_universe_survivorship_corrected.txt"
SCREEN_DATES = [date(2017, 6, 30), date(2020, 6, 30), date(2023, 6, 30), date(2025, 12, 31)]


def main() -> int:
    load_dotenv()
    client = StockHistoricalDataClient(
        os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]
    )
    master = [s for s in MASTER.read_text().split() if s]
    print(f"Master universe: {len(master):,} symbols")
    print(f"Screening liquidity at {len(SCREEN_DATES)} dates...")
    universe = build_union_universe(client, master, SCREEN_DATES, ScreenConfig(), progress=True)
    OUT.write_text("\n".join(universe))
    print(f"\nSurvivorship-corrected investable universe: {len(universe):,} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
