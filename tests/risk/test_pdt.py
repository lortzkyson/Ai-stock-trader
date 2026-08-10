from __future__ import annotations

from datetime import date, timedelta

from risk.pdt import DayTradeTracker


def _weekdays(start: date, n: int) -> list[date]:
    days = []
    d = start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def test_blocks_fourth_day_trade_within_window_under_25k() -> None:
    calendar = _weekdays(date(2026, 1, 5), 5)
    tracker = DayTradeTracker(equity_threshold=25_000.0)

    for d in calendar[:3]:
        assert tracker.can_open_day_trade(d, calendar, account_equity=10_000)
        tracker.record_day_trade(d)

    assert not tracker.can_open_day_trade(calendar[3], calendar, account_equity=10_000)


def test_unrestricted_above_equity_threshold() -> None:
    calendar = _weekdays(date(2026, 1, 5), 5)
    tracker = DayTradeTracker(equity_threshold=25_000.0)

    for d in calendar:
        tracker.record_day_trade(d)

    assert tracker.can_open_day_trade(calendar[-1], calendar, account_equity=30_000)


def test_window_rolls_off_old_day_trades() -> None:
    calendar = _weekdays(date(2026, 1, 5), 10)
    tracker = DayTradeTracker(equity_threshold=25_000.0)

    for d in calendar[:3]:
        tracker.record_day_trade(d)

    # Far enough ahead that the first 3 day trades have rolled out of the
    # 5-business-day window, so a new one should be allowed again.
    assert tracker.can_open_day_trade(calendar[8], calendar, account_equity=10_000)
