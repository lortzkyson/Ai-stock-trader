#!/usr/bin/env python3
"""Pull, clean, and cache the training dataset for Phases 3-5.

Widened from the initial 8-symbol/~8-month build after Phase 4/5 found no
edge over a random baseline: full 30-symbol seed universe, ~14 months of
1-minute bars ending right before the pinned holdout window. Widen further
via SYMBOLS/START/END below if needed; nothing else needs to change.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.alpaca_client import AlpacaBarsClient, BarsQuery  # noqa: E402
from data.holdout import exclude_holdout  # noqa: E402
from data.pipeline import run_pipeline  # noqa: E402
from data.quality import check_quality  # noqa: E402
from data.universe import load_seed_universe  # noqa: E402

SYMBOLS = load_seed_universe()
START = date(2025, 3, 1)
END = date(2026, 4, 30)  # holdout starts 2026-05-01 — see docs/pre-mortem.md


def main() -> int:
    load_dotenv()
    client = AlpacaBarsClient()

    def fetch_fn(symbol: str, start: date, end: date):
        query = BarsQuery(
            symbols=[symbol],
            start=datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc),
            end=datetime.combine(end, datetime.min.time(), tzinfo=timezone.utc),
        )
        return client.fetch_bars(query)

    result = run_pipeline(SYMBOLS, START, END, fetch_fn=fetch_fn)

    for symbol, bars in result.bars.items():
        bars = exclude_holdout(bars)
        report = check_quality(bars, symbol)
        print(
            f"{symbol}: {len(bars)} clean regular-session rows, "
            f"is_clean={report.is_clean}, halt_like_days={len(report.halt_like_days)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
