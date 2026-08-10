"""Pattern Day Trader (PDT) rule tracking.

FINRA: a "day trade" is opening and closing the same position on the same
trading day. Margin accounts under $25k equity are limited to 3 day trades
in any rolling 5-business-day window — violating this risks a trading
restriction/freeze (docs/pre-mortem.md, "PDT and strategy shape"). This is
why the strategy is a hybrid of day trades and multi-day swing holds rather
than pure intraday: this tracker blocks a new same-day round trip once the
budget is spent, but a position can still be opened and carried past the
same session.
"""

from __future__ import annotations

from bisect import bisect_right
from datetime import date

MAX_DAY_TRADES_PER_5_DAYS = 3
WINDOW_SIZE = 5


def _last_n_trading_days(trading_calendar: list[date], as_of_date: date, n: int) -> set[date]:
    calendar = sorted(trading_calendar)
    idx = bisect_right(calendar, as_of_date) - 1
    if idx < 0:
        return set()
    start = max(0, idx - n + 1)
    return set(calendar[start : idx + 1])


class DayTradeTracker:
    def __init__(self, equity_threshold: float = 25_000.0) -> None:
        self.equity_threshold = equity_threshold
        self._day_trade_dates: list[date] = []

    def record_day_trade(self, trade_date: date) -> None:
        self._day_trade_dates.append(trade_date)

    def day_trade_count_in_window(self, as_of_date: date, trading_calendar: list[date]) -> int:
        window = _last_n_trading_days(trading_calendar, as_of_date, WINDOW_SIZE)
        return sum(1 for d in self._day_trade_dates if d in window)

    def can_open_day_trade(
        self, as_of_date: date, trading_calendar: list[date], account_equity: float
    ) -> bool:
        if account_equity >= self.equity_threshold:
            return True
        count = self.day_trade_count_in_window(as_of_date, trading_calendar)
        return count < MAX_DAY_TRADES_PER_5_DAYS
