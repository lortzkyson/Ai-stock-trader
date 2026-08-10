"""Reconciliation: compare internal expected state against Alpaca's actual
positions. Run on startup and periodically — flag any mismatch loudly
rather than silently trusting local state. A dropped connection or a missed
fill notification could otherwise leave the system tracking phantom
positions indefinitely.
"""

from __future__ import annotations

from dataclasses import dataclass

from execution.client import AlpacaExecutionClient
from execution.structured_logging import StructuredLogger


@dataclass
class ReconciliationResult:
    is_consistent: bool
    mismatches: list[str]


def reconcile_positions(
    expected_positions: dict[str, int],
    client: AlpacaExecutionClient,
    logger: StructuredLogger,
) -> ReconciliationResult:
    actual = client.get_positions()
    actual_positions = {p.symbol: int(float(p.qty)) for p in actual}

    mismatches = []
    for symbol in sorted(set(expected_positions) | set(actual_positions)):
        expected_qty = expected_positions.get(symbol, 0)
        actual_qty = actual_positions.get(symbol, 0)
        if expected_qty != actual_qty:
            msg = f"{symbol}: expected {expected_qty} shares, Alpaca reports {actual_qty}"
            mismatches.append(msg)
            logger.log_reconciliation_mismatch(
                msg, symbol=symbol, expected=expected_qty, actual=actual_qty
            )

    return ReconciliationResult(is_consistent=not mismatches, mismatches=mismatches)
