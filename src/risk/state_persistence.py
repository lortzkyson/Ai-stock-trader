"""Persist RiskEngine state to disk across scheduled loop invocations.

Each run of scripts/paper_trading_loop.py is a short-lived process — day-trade
history, the drawdown breaker's peak equity, and the daily loss limit's
day-start equity all need to survive between invocations, not reset every
time the scheduler fires the script.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from risk.engine import RiskEngine

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_STATE_PATH = REPO_ROOT / "data" / "risk_state.json"


def save_risk_state(risk_engine: RiskEngine, path: Path = DEFAULT_STATE_PATH) -> None:
    state = {
        "day_trade_dates": [d.isoformat() for d in risk_engine.day_trade_tracker._day_trade_dates],
        "drawdown_peak_equity": risk_engine.drawdown_breaker._peak_equity,
        "drawdown_tripped": risk_engine.drawdown_breaker._tripped,
        "daily_loss_day": risk_engine.daily_loss_limit._day.isoformat()
        if risk_engine.daily_loss_limit._day
        else None,
        "daily_loss_day_start_equity": risk_engine.daily_loss_limit._day_start_equity,
        "daily_loss_tripped": risk_engine.daily_loss_limit._tripped,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def load_risk_state(risk_engine: RiskEngine, path: Path = DEFAULT_STATE_PATH) -> None:
    if not path.exists():
        return
    state = json.loads(path.read_text())

    risk_engine.day_trade_tracker._day_trade_dates = [
        date.fromisoformat(d) for d in state.get("day_trade_dates", [])
    ]
    risk_engine.drawdown_breaker._peak_equity = state.get("drawdown_peak_equity", 0.0)
    risk_engine.drawdown_breaker._tripped = state.get("drawdown_tripped", False)
    daily_loss_day = state.get("daily_loss_day")
    risk_engine.daily_loss_limit._day = (
        date.fromisoformat(daily_loss_day) if daily_loss_day else None
    )
    risk_engine.daily_loss_limit._day_start_equity = state.get("daily_loss_day_start_equity")
    risk_engine.daily_loss_limit._tripped = state.get("daily_loss_tripped", False)
