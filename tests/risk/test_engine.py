from __future__ import annotations

from datetime import date, timedelta

import pytest

from features.labeling import TripleBarrierConfig
from risk.engine import RiskConfig, RiskEngine


def _weekdays(start: date, n: int) -> list[date]:
    days = []
    d = start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def test_daily_loss_limit_stops_trading_after_a_bad_day() -> None:
    config = RiskConfig(daily_loss_limit_pct=0.02, max_drawdown_pct=0.5)
    engine = RiskEngine(config)
    d = date(2026, 1, 5)
    calendar = [d]

    engine.update_equity(d, 100_000)
    decision = engine.evaluate_entry(
        d, entry_price=100.0, equity=100_000, trading_calendar=calendar, is_intended_day_trade=False
    )
    assert decision.approved

    engine.update_equity(d, 97_000)  # -3%, past the 2% daily limit
    decision = engine.evaluate_entry(
        d, entry_price=100.0, equity=97_000, trading_calendar=calendar, is_intended_day_trade=False
    )
    assert not decision.approved
    assert decision.reason == "daily_loss_limit_tripped"


def test_losing_streak_trips_max_drawdown_breaker_and_blocks_future_trades() -> None:
    """Simulate a sustained losing streak and confirm the drawdown circuit
    breaker actually halts new trades — the explicit check the risk-module
    spec calls for."""
    # Daily limit deliberately loose so the drawdown breaker is what trips.
    config = RiskConfig(daily_loss_limit_pct=0.5, max_drawdown_pct=0.10)
    engine = RiskEngine(config)
    calendar = _weekdays(date(2026, 1, 5), 30)

    equity = 100_000.0
    approvals = []
    for d in calendar:
        engine.update_equity(d, equity)
        decision = engine.evaluate_entry(
            d, entry_price=100.0, equity=equity, trading_calendar=calendar,
            is_intended_day_trade=False,
        )
        approvals.append(decision.approved)
        if decision.approved:
            equity *= 0.99  # every approved trade loses 1%

    assert approvals[0] is True
    assert approvals[-1] is False
    assert engine.drawdown_breaker.is_tripped

    # Stays halted even if equity partially recovers — no auto-reset.
    engine.update_equity(calendar[-1], equity * 1.5)
    decision = engine.evaluate_entry(
        calendar[-1], entry_price=100.0, equity=equity * 1.5, trading_calendar=calendar,
        is_intended_day_trade=False,
    )
    assert not decision.approved

    engine.drawdown_breaker.manual_reset()
    decision = engine.evaluate_entry(
        calendar[-1], entry_price=100.0, equity=equity * 1.5, trading_calendar=calendar,
        is_intended_day_trade=False,
    )
    assert decision.approved


def test_pdt_blocks_fourth_intraday_round_trip_under_25k() -> None:
    config = RiskConfig(
        pdt_equity_threshold=25_000.0, daily_loss_limit_pct=0.5, max_drawdown_pct=0.5
    )
    engine = RiskEngine(config)
    calendar = _weekdays(date(2026, 1, 5), 5)
    equity = 10_000.0

    for d in calendar[:3]:
        engine.update_equity(d, equity)
        decision = engine.evaluate_entry(
            d, entry_price=50.0, equity=equity, trading_calendar=calendar,
            is_intended_day_trade=True,
        )
        assert decision.approved
        # evaluate_entry only checks/sizes — recording is an explicit, separate
        # step the caller does once it knows the round trip actually closed
        # same-day (see risk/engine.py's evaluate_entry docstring).
        engine.day_trade_tracker.record_day_trade(d)

    engine.update_equity(calendar[3], equity)
    decision = engine.evaluate_entry(
        calendar[3], entry_price=50.0, equity=equity, trading_calendar=calendar,
        is_intended_day_trade=True,
    )
    assert not decision.approved
    assert decision.reason == "pdt_limit_would_be_violated"

    # A multi-day swing entry (not a same-day round trip) should still be allowed.
    decision = engine.evaluate_entry(
        calendar[3], entry_price=50.0, equity=equity, trading_calendar=calendar,
        is_intended_day_trade=False,
    )
    assert decision.approved


def test_stop_and_target_prices_come_from_barrier_config() -> None:
    barrier = TripleBarrierConfig(profit_target_pct=0.03, stop_loss_pct=0.015, max_holding_bars=100)
    config = RiskConfig(barrier_config=barrier, daily_loss_limit_pct=0.5, max_drawdown_pct=0.5)
    engine = RiskEngine(config)
    d = date(2026, 1, 5)
    engine.update_equity(d, 100_000)

    decision = engine.evaluate_entry(
        d, entry_price=200.0, equity=100_000, trading_calendar=[d], is_intended_day_trade=False
    )

    assert decision.approved
    assert decision.stop_loss_price == pytest.approx(200 * (1 - 0.015))
    assert decision.take_profit_price == pytest.approx(200 * (1 + 0.03))


def test_gross_exposure_cap_blocks_accidental_leverage() -> None:
    """Regression: per-position limits don't compose. With
    max_position_fraction=0.25, eight individually-legal positions sum to 200%
    of equity. The live paper account reached 1.99x gross leverage on margin
    this way, with every sizing check passing."""
    config = RiskConfig(
        max_position_fraction=0.25, max_gross_exposure=1.0,
        daily_loss_limit_pct=0.5, max_drawdown_pct=0.5,
    )
    engine = RiskEngine(config)
    d = date(2026, 1, 5)
    equity = 100_000.0
    engine.update_equity(d, equity)

    gross = 0.0
    approved = 0
    for _ in range(12):
        decision = engine.evaluate_entry(
            d, entry_price=100.0, equity=equity, trading_calendar=[d],
            is_intended_day_trade=False, current_gross_exposure=gross,
        )
        if not decision.approved:
            break
        approved += 1
        gross += decision.shares * 100.0

    assert gross <= equity * 1.0 + 1e-6, f"gross exposure {gross} exceeded equity"
    assert approved < 12, "should have refused before deploying 12 full positions"


def test_gross_exposure_cap_trims_the_last_position_to_fit() -> None:
    config = RiskConfig(
        max_position_fraction=0.5, max_gross_exposure=1.0,
        daily_loss_limit_pct=0.5, max_drawdown_pct=0.5,
    )
    engine = RiskEngine(config)
    d = date(2026, 1, 5)
    engine.update_equity(d, 100_000.0)

    # 90% already deployed: only $10,000 of headroom remains.
    decision = engine.evaluate_entry(
        d, entry_price=100.0, equity=100_000.0, trading_calendar=[d],
        is_intended_day_trade=False, current_gross_exposure=90_000.0,
    )

    assert decision.approved
    assert decision.shares * 100.0 <= 10_000 + 1e-6


def test_gross_exposure_cap_refuses_when_fully_deployed() -> None:
    config = RiskConfig(max_gross_exposure=1.0, daily_loss_limit_pct=0.5, max_drawdown_pct=0.5)
    engine = RiskEngine(config)
    d = date(2026, 1, 5)
    engine.update_equity(d, 100_000.0)

    decision = engine.evaluate_entry(
        d, entry_price=100.0, equity=100_000.0, trading_calendar=[d],
        is_intended_day_trade=False, current_gross_exposure=100_000.0,
    )

    assert not decision.approved
    assert decision.reason == "max_gross_exposure_reached"


def test_leverage_is_allowed_only_when_explicitly_configured() -> None:
    """Leverage should be a deliberate setting, never an emergent accident."""
    levered = RiskConfig(max_gross_exposure=2.0, daily_loss_limit_pct=0.5, max_drawdown_pct=0.5)
    engine = RiskEngine(levered)
    d = date(2026, 1, 5)
    engine.update_equity(d, 100_000.0)

    decision = engine.evaluate_entry(
        d, entry_price=100.0, equity=100_000.0, trading_calendar=[d],
        is_intended_day_trade=False, current_gross_exposure=150_000.0,
    )

    assert decision.approved, "explicit 2x config should permit exposure above 1x"
