"""Structured (JSON-lines) logging of every signal, order, and fill.

One JSON object per line, appended to a log file — easy to grep, easy to
load into pandas later for analysis, and a plain append avoids any
partial-write corruption risk from concurrent processes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LOG_PATH = REPO_ROOT / "data" / "execution_log.jsonl"


class StructuredLogger:
    def __init__(self, log_path: Path = DEFAULT_LOG_PATH) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, **fields: Any) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            **fields,
        }
        with self.log_path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def log_signal(self, symbol: str, proba: float, predicted: int) -> None:
        self.log("signal", symbol=symbol, proba=proba, predicted=predicted)

    def log_order_submitted(
        self, symbol: str, side: str, qty: int, order_type: str, order_id: str
    ) -> None:
        self.log(
            "order_submitted", symbol=symbol, side=side, qty=qty,
            order_type=order_type, order_id=order_id,
        )

    def log_fill(self, symbol: str, side: str, qty: int, price: float, order_id: str) -> None:
        self.log("fill", symbol=symbol, side=side, qty=qty, price=price, order_id=order_id)

    def log_reconciliation_mismatch(self, description: str, **details: Any) -> None:
        self.log("reconciliation_mismatch", description=description, **details)

    def log_kill_switch(self, action: str) -> None:
        self.log("kill_switch", action=action)
