"""Persist locally-tracked open-position state (entry price, stop/target
levels, max-holding deadline) across loop iterations.

Alpaca's own position objects don't carry *our* stop-loss/take-profit levels
or entry date — those only exist in this local state, keyed by symbol.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_POSITION_STATE_PATH = REPO_ROOT / "data" / "open_positions_state.json"


def load_open_positions(path: Path = DEFAULT_POSITION_STATE_PATH) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    result: dict[str, dict[str, Any]] = json.loads(path.read_text())
    return result


def save_open_positions(
    positions: dict[str, dict[str, Any]], path: Path = DEFAULT_POSITION_STATE_PATH
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(positions, indent=2))
