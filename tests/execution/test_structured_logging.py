from __future__ import annotations

import json
from pathlib import Path

from execution.structured_logging import StructuredLogger


def test_log_writes_one_json_line_per_event(tmp_path: Path) -> None:
    logger = StructuredLogger(log_path=tmp_path / "log.jsonl")

    logger.log_signal("AAPL", proba=0.7, predicted=1)
    logger.log_order_submitted("AAPL", side="buy", qty=10, order_type="market", order_id="abc123")
    logger.log_fill("AAPL", side="buy", qty=10, price=150.25, order_id="abc123")

    lines = (tmp_path / "log.jsonl").read_text().splitlines()
    assert len(lines) == 3

    records = [json.loads(line) for line in lines]
    assert records[0]["event_type"] == "signal"
    assert records[0]["symbol"] == "AAPL"
    assert records[1]["event_type"] == "order_submitted"
    assert records[2]["event_type"] == "fill"
    assert records[2]["price"] == 150.25
    assert "timestamp" in records[0]
