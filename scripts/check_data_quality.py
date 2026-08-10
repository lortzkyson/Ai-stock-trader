#!/usr/bin/env python3
"""CLI: run data-quality checks against one or more cached parquet bar files.

Usage:
    scripts/check_data_quality.py data/cache/1Min/*.parquet
    scripts/check_data_quality.py data/cache/1Min/AAPL_2026-01-01_2026-03-01.parquet

Exits non-zero if any file fails the quality check, so it can be wired into
CI later without extra glue.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.quality import check_quality  # noqa: E402


def symbol_from_filename(path: Path) -> str:
    return path.stem.split("_")[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path, help="Parquet bar files to check")
    args = parser.parse_args(argv)

    any_dirty = False
    for path in args.files:
        bars = pd.read_parquet(path)
        symbol = symbol_from_filename(path)
        report = check_quality(bars, symbol)

        status = "OK" if report.is_clean else "FLAGGED"
        print(f"[{status}] {path.name}")
        print(f"  duplicate timestamps: {report.duplicate_timestamp_count}")
        print(f"  impossible values:    {report.impossible_value_count}")
        print(f"  extreme jumps:        {report.extreme_jump_count} (flagged, not auto-excluded)")
        print(f"  halt-like days:       {len(report.halt_like_days)} {report.halt_like_days}")

        if not report.is_clean:
            any_dirty = True

    return 1 if any_dirty else 0


if __name__ == "__main__":
    raise SystemExit(main())
