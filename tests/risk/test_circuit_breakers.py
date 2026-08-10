from __future__ import annotations

from datetime import date

from risk.circuit_breakers import DailyLossLimit, MaxDrawdownBreaker


def test_daily_loss_limit_trips_within_the_day() -> None:
    limit = DailyLossLimit(limit_pct=0.02)
    d = date(2026, 1, 5)

    limit.update(d, 100_000)
    assert not limit.is_tripped

    limit.update(d, 97_500)  # -2.5%, past the 2% limit
    assert limit.is_tripped


def test_daily_loss_limit_resets_next_day() -> None:
    limit = DailyLossLimit(limit_pct=0.02)
    day1, day2 = date(2026, 1, 5), date(2026, 1, 6)

    limit.update(day1, 100_000)
    limit.update(day1, 97_000)
    assert limit.is_tripped

    limit.update(day2, 97_000)  # new day, resets against the new day's own start
    assert not limit.is_tripped


def test_max_drawdown_breaker_trips_from_peak() -> None:
    breaker = MaxDrawdownBreaker(limit_pct=0.10)

    breaker.update(100_000)
    breaker.update(110_000)  # new peak
    assert not breaker.is_tripped

    breaker.update(98_000)  # -10.9% from peak of 110k
    assert breaker.is_tripped


def test_max_drawdown_breaker_does_not_auto_reset() -> None:
    breaker = MaxDrawdownBreaker(limit_pct=0.10)
    breaker.update(100_000)
    breaker.update(85_000)
    assert breaker.is_tripped

    breaker.update(120_000)  # equity recovers well past the old peak
    assert breaker.is_tripped  # still tripped — needs an explicit manual_reset()

    breaker.manual_reset()
    assert not breaker.is_tripped
