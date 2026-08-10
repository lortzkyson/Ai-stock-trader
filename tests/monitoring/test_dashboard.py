from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from monitoring.dashboard import render_dashboard
from risk.pdt import DayTradeTracker


@dataclass
class FakePosition:
    symbol: str
    qty: str
    unrealized_pl: str


class FakeClient:
    def __init__(self, equity: float, positions: list[FakePosition]) -> None:
        self._equity = equity
        self._positions = positions

    def get_account_equity(self) -> float:
        return self._equity

    def get_positions(self) -> list[FakePosition]:
        return self._positions


def _weekdays(start: date, n: int) -> list[date]:
    days = []
    d = start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def test_render_dashboard_shows_equity_and_positions() -> None:
    client = FakeClient(12_345.67, [FakePosition("AAPL", "10", "42.50")])
    tracker = DayTradeTracker()
    calendar = _weekdays(date(2026, 1, 5), 5)

    output = render_dashboard(client, tracker, calendar, as_of=calendar[0])  # type: ignore[arg-type]

    assert "$12,345.67" in output
    assert "AAPL" in output
    assert "Day trades" in output


def test_render_dashboard_flags_when_pdt_budget_is_spent() -> None:
    client = FakeClient(10_000.0, [])
    tracker = DayTradeTracker()
    calendar = _weekdays(date(2026, 1, 5), 5)
    for d in calendar[:3]:
        tracker.record_day_trade(d)

    output = render_dashboard(client, tracker, calendar, as_of=calendar[2])  # type: ignore[arg-type]

    assert "3/3" in output
    assert "PDT budget spent" in output
