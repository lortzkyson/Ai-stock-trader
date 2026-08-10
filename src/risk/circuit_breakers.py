"""Daily loss limit and max-drawdown circuit breakers.

Both are hard stops: once tripped, the risk engine refuses new trades. The
daily limit resets automatically at the start of the next trading day. The
drawdown breaker deliberately does not auto-reset — it requires an explicit
`manual_reset()` call, standing in for the human review the spec calls for
before trading resumes after a large drawdown.
"""

from __future__ import annotations

from datetime import date


class DailyLossLimit:
    def __init__(self, limit_pct: float) -> None:
        self.limit_pct = limit_pct
        self._day: date | None = None
        self._day_start_equity: float | None = None
        self._tripped = False

    def update(self, current_date: date, current_equity: float) -> None:
        if self._day != current_date:
            self._day = current_date
            self._day_start_equity = current_equity
            self._tripped = False

        if self._day_start_equity and current_equity <= self._day_start_equity * (
            1 - self.limit_pct
        ):
            self._tripped = True

    @property
    def is_tripped(self) -> bool:
        return self._tripped


class MaxDrawdownBreaker:
    def __init__(self, limit_pct: float) -> None:
        self.limit_pct = limit_pct
        self._peak_equity = 0.0
        self._tripped = False

    def update(self, current_equity: float) -> None:
        self._peak_equity = max(self._peak_equity, current_equity)
        if self._peak_equity > 0 and current_equity <= self._peak_equity * (1 - self.limit_pct):
            self._tripped = True

    @property
    def is_tripped(self) -> bool:
        return self._tripped

    def manual_reset(self) -> None:
        """Explicit reset after human review — this breaker never auto-resets."""
        self._tripped = False
        self._peak_equity = 0.0
