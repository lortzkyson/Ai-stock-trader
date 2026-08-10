#!/usr/bin/env python3
"""Manual kill switch — independent of the automatic circuit breakers in src/risk/.

Usage:
    scripts/kill_switch.py status
    scripts/kill_switch.py engage [--flatten]
    scripts/kill_switch.py disengage
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from execution import kill_switch  # noqa: E402
from execution.structured_logging import StructuredLogger  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["status", "engage", "disengage"])
    parser.add_argument(
        "--flatten", action="store_true", help="also close all open positions (engage only)"
    )
    args = parser.parse_args()
    logger = StructuredLogger()

    if args.action == "status":
        print("ENGAGED" if kill_switch.is_engaged() else "not engaged")
        return 0

    if args.action == "disengage":
        kill_switch.disengage()
        logger.log_kill_switch("disengaged")
        print("Kill switch disengaged. Trading may resume.")
        return 0

    kill_switch.engage()
    logger.log_kill_switch("engaged")
    print("Kill switch ENGAGED. No new orders will be submitted.")

    if args.flatten:
        from dotenv import load_dotenv

        load_dotenv()
        from execution.client import AlpacaExecutionClient

        client = AlpacaExecutionClient()
        client.close_all_positions()
        logger.log_kill_switch("flattened")
        print("All positions flattened.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
