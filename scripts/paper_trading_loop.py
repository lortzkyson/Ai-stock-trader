#!/usr/bin/env python3
"""One iteration of the paper-trading loop. Meant to be invoked on a
schedule (e.g. every 5 minutes during regular trading hours) — see
docs/runbook.md. Does not loop or sleep itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.universe import load_seed_universe  # noqa: E402
from execution.loop import run_iteration  # noqa: E402


def main() -> int:
    load_dotenv()
    universe = load_seed_universe()
    run_iteration(universe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
