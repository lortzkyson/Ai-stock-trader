"""Append-only experiment log (docs/pre-mortem.md §5) — one row per training/backtest run.

The point is to make the iteration process itself auditable: if a final
result looks too good, this log lets us check how many variations were
tried against the same data to get there.
"""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXPERIMENT_LOG_PATH = REPO_ROOT / "experiment_log.csv"

FIELDS = [
    "run_id",
    "date",
    "git_commit",
    "phase",
    "data_start",
    "data_end",
    "params_json",
    "win_rate",
    "expectancy",
    "sharpe",
    "sortino",
    "max_drawdown",
    "notes",
]


def current_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def append_run(
    phase: str,
    data_start: str,
    data_end: str,
    params: dict,
    win_rate: float,
    expectancy: float,
    sharpe: float,
    sortino: float = float("nan"),
    max_drawdown: float = float("nan"),
    notes: str = "",
    log_path: Path = EXPERIMENT_LOG_PATH,
) -> None:
    exists = log_path.exists()
    run_id = 1
    if exists:
        with log_path.open() as f:
            # Header line counted -> next run_id == existing row count + 1.
            run_id = sum(1 for _ in f)

    with log_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "run_id": run_id,
                "date": datetime.now(timezone.utc).isoformat(),
                "git_commit": current_git_commit(),
                "phase": phase,
                "data_start": data_start,
                "data_end": data_end,
                "params_json": json.dumps(params),
                "win_rate": win_rate,
                "expectancy": expectancy,
                "sharpe": sharpe,
                "sortino": sortino,
                "max_drawdown": max_drawdown,
                "notes": notes,
            }
        )
