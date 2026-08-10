from __future__ import annotations

import csv
from pathlib import Path

from models.experiment_log import append_run


def test_append_run_writes_header_once_and_increments_run_id(tmp_path: Path) -> None:
    log_path = tmp_path / "experiment_log.csv"

    append_run(
        phase="test",
        data_start="2026-01-01",
        data_end="2026-01-31",
        params={"n_folds": 5},
        win_rate=0.4,
        expectancy=0.001,
        sharpe=0.5,
        log_path=log_path,
    )
    append_run(
        phase="test",
        data_start="2026-01-01",
        data_end="2026-01-31",
        params={"n_folds": 6},
        win_rate=0.42,
        expectancy=0.0015,
        sharpe=0.6,
        log_path=log_path,
    )

    with log_path.open() as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]["run_id"] == "1"
    assert rows[1]["run_id"] == "2"
    assert rows[1]["params_json"] == '{"n_folds": 6}'
