#!/usr/bin/env python3
"""Print the live monitoring dashboard: equity, open positions, day-trade
count vs. the PDT limit. See docs/runbook.md for what to do with what you see.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from execution.client import AlpacaExecutionClient  # noqa: E402
from monitoring.dashboard import render_dashboard  # noqa: E402
from risk.pdt import DayTradeTracker  # noqa: E402


def _recent_weekdays(n: int = 10) -> list[date]:
    days = []
    d = date.today()
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    return sorted(days)


def main() -> int:
    load_dotenv()
    client = AlpacaExecutionClient()
    # NOTE: this tracker starts empty each run — it doesn't persist day-trade
    # history across process restarts. A real deployment needs to load this
    # from the structured execution log (data/execution_log.jsonl) or a
    # small local store; flagged here rather than solved, since this script
    # is a one-shot status check, not the live trading loop itself.
    tracker = DayTradeTracker()
    print(render_dashboard(client, tracker, _recent_weekdays()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
