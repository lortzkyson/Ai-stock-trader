"""Manual kill switch: a plain flag file that halts new order submission,
independent of the automatic circuit breakers in src/risk/. Meant to be
triggered by hand the moment something looks wrong — see scripts/kill_switch.py.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
KILL_SWITCH_FLAG = REPO_ROOT / "data" / "KILL_SWITCH_ENGAGED"


def engage(flag_path: Path = KILL_SWITCH_FLAG) -> None:
    flag_path.parent.mkdir(parents=True, exist_ok=True)
    flag_path.write_text("kill switch engaged\n")


def disengage(flag_path: Path = KILL_SWITCH_FLAG) -> None:
    flag_path.unlink(missing_ok=True)


def is_engaged(flag_path: Path = KILL_SWITCH_FLAG) -> bool:
    return flag_path.exists()
