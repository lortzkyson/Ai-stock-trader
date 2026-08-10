from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from execution.reconciliation import reconcile_positions
from execution.structured_logging import StructuredLogger


@dataclass
class FakePosition:
    symbol: str
    qty: str


class FakeClient:
    def __init__(self, positions: list[FakePosition]) -> None:
        self._positions = positions

    def get_positions(self) -> list[FakePosition]:
        return self._positions


def test_reconcile_positions_matching_state_is_consistent(tmp_path: Path) -> None:
    client = FakeClient([FakePosition("AAPL", "10")])
    logger = StructuredLogger(log_path=tmp_path / "log.jsonl")

    result = reconcile_positions({"AAPL": 10}, client, logger)  # type: ignore[arg-type]

    assert result.is_consistent
    assert result.mismatches == []


def test_reconcile_positions_flags_quantity_mismatch(tmp_path: Path) -> None:
    client = FakeClient([FakePosition("AAPL", "5")])
    logger = StructuredLogger(log_path=tmp_path / "log.jsonl")

    result = reconcile_positions({"AAPL": 10}, client, logger)  # type: ignore[arg-type]

    assert not result.is_consistent
    assert "AAPL" in result.mismatches[0]
    assert (tmp_path / "log.jsonl").exists()


def test_reconcile_positions_flags_phantom_local_position(tmp_path: Path) -> None:
    client = FakeClient([])  # Alpaca reports nothing open
    logger = StructuredLogger(log_path=tmp_path / "log.jsonl")

    result = reconcile_positions({"AAPL": 10}, client, logger)  # type: ignore[arg-type]

    assert not result.is_consistent


def test_reconcile_positions_flags_untracked_alpaca_position(tmp_path: Path) -> None:
    client = FakeClient([FakePosition("MSFT", "3")])  # Alpaca has a position we don't know about
    logger = StructuredLogger(log_path=tmp_path / "log.jsonl")

    result = reconcile_positions({}, client, logger)  # type: ignore[arg-type]

    assert not result.is_consistent
