from __future__ import annotations

from pathlib import Path

from execution.position_state import load_open_positions, save_open_positions


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "positions.json"
    positions = {
        "AAPL": {
            "entry_price": 150.0,
            "stop_loss_price": 148.5,
            "take_profit_price": 153.0,
            "entry_date": "2026-01-05",
            "max_holding_deadline": "2026-01-08T16:00:00+00:00",
        }
    }

    save_open_positions(positions, path)
    loaded = load_open_positions(path)

    assert loaded == positions


def test_load_missing_file_returns_empty_dict(tmp_path: Path) -> None:
    assert load_open_positions(tmp_path / "nope.json") == {}
