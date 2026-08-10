#!/usr/bin/env python3
"""Compare live/paper performance against the latest backtest report and flag
divergence. Meant to be run on a schedule (daily/weekly) once real fills
exist in data/execution_log.jsonl — see docs/runbook.md.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from execution.structured_logging import DEFAULT_LOG_PATH  # noqa: E402
from monitoring.performance_report import build_live_metrics, compare_to_baseline  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
STARTING_EQUITY = 10_000.0  # keep in sync with scripts/run_backtest.py

# Baseline: the most recent backtest run's metrics. Update this once real
# paper/live data justifies a real comparison; for now this mirrors
# reports/backtest_2026-08-10.md so the script is runnable end-to-end.
BACKTEST_BASELINE = {
    "win_rate": 0.3478,
    "expectancy": -0.0004,
    "sharpe": -0.2969,
    "max_drawdown": -0.0692,
}


def main() -> int:
    if not DEFAULT_LOG_PATH.exists():
        print(f"No execution log at {DEFAULT_LOG_PATH} yet — nothing to compare.")
        return 0

    live_metrics = build_live_metrics(DEFAULT_LOG_PATH, STARTING_EQUITY)
    print("Live metrics:", live_metrics)

    if live_metrics["n_trades"] == 0:
        print("No completed round-trip trades in the log yet.")
        return 0

    flags = compare_to_baseline(live_metrics, BACKTEST_BASELINE)
    report_path = REPO_ROOT / "reports" / f"performance_{datetime.now(timezone.utc):%Y-%m-%d}.md"
    lines = [f"# Performance Report ({datetime.now(timezone.utc):%Y-%m-%d})\n"]
    lines.append(f"Live trades reconstructed from fills: {live_metrics['n_trades']}\n")
    lines.append("| metric | live | backtest baseline |")
    lines.append("|---|---|---|")
    for k in ["win_rate", "expectancy", "sharpe", "max_drawdown"]:
        lines.append(f"| {k} | {live_metrics[k]:.4f} | {BACKTEST_BASELINE[k]:.4f} |")
    lines.append("")
    if flags:
        lines.append("## Flagged divergence\n")
        for flag in flags:
            lines.append(f"- {flag.message}")
    else:
        lines.append("No metric diverged from the backtest baseline beyond the configured tolerance.")
    report_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
