from __future__ import annotations

from datetime import date
from pathlib import Path

from risk.engine import RiskConfig, RiskEngine
from risk.state_persistence import load_risk_state, save_risk_state


def test_save_then_load_round_trips_state(tmp_path: Path) -> None:
    path = tmp_path / "risk_state.json"
    engine = RiskEngine(RiskConfig())
    engine.day_trade_tracker.record_day_trade(date(2026, 1, 5))
    engine.day_trade_tracker.record_day_trade(date(2026, 1, 6))
    engine.update_equity(date(2026, 1, 6), 100_000)
    engine.update_equity(date(2026, 1, 6), 97_000)  # trips daily loss limit
    engine.drawdown_breaker.update(80_000)  # trips drawdown breaker too

    save_risk_state(engine, path)

    restored = RiskEngine(RiskConfig())
    load_risk_state(restored, path)

    assert restored.day_trade_tracker._day_trade_dates == [date(2026, 1, 5), date(2026, 1, 6)]
    assert restored.daily_loss_limit.is_tripped
    assert restored.drawdown_breaker.is_tripped
    assert restored.drawdown_breaker._peak_equity == engine.drawdown_breaker._peak_equity


def test_load_risk_state_missing_file_is_a_noop(tmp_path: Path) -> None:
    path = tmp_path / "does_not_exist.json"
    engine = RiskEngine(RiskConfig())
    load_risk_state(engine, path)  # should not raise
    assert engine.day_trade_tracker._day_trade_dates == []
