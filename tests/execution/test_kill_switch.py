from __future__ import annotations

from pathlib import Path

from execution import kill_switch


def test_engage_disengage_round_trip(tmp_path: Path) -> None:
    flag = tmp_path / "KILL_SWITCH_ENGAGED"

    assert not kill_switch.is_engaged(flag)

    kill_switch.engage(flag)
    assert kill_switch.is_engaged(flag)

    kill_switch.disengage(flag)
    assert not kill_switch.is_engaged(flag)


def test_disengage_when_not_engaged_is_a_noop(tmp_path: Path) -> None:
    flag = tmp_path / "KILL_SWITCH_ENGAGED"
    kill_switch.disengage(flag)  # should not raise
    assert not kill_switch.is_engaged(flag)
