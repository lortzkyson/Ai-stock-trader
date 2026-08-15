"""Standalone risk engine used by both the backtester (Phase 5) and live
execution (Phase 7) — risk logic must not live only in the backtest
(docs/pre-mortem.md guard #9). Stop-loss/take-profit levels come from the
same TripleBarrierConfig used to label training data, so backtest and live
behavior match.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from features.labeling import TripleBarrierConfig
from risk.circuit_breakers import DailyLossLimit, MaxDrawdownBreaker
from risk.pdt import DayTradeTracker
from risk.position_sizing import fixed_fractional_shares


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade_pct: float = 0.01
    max_position_fraction: float = 0.25
    daily_loss_limit_pct: float = 0.02
    max_drawdown_pct: float = 0.10
    pdt_equity_threshold: float = 25_000.0
    barrier_config: TripleBarrierConfig = field(default_factory=TripleBarrierConfig)

    # Ceiling on TOTAL deployed capital as a fraction of equity. 1.0 = never
    # borrow. This exists because per-position limits do not compose: with
    # max_position_fraction=0.25, eight individually-legal positions sum to 200%
    # of equity. That is exactly what happened on the live paper account — it
    # reached 1.99x gross leverage on margin with every single sizing check
    # passing, because nothing was checking the aggregate.
    max_gross_exposure: float = 1.0


@dataclass
class TradeDecision:
    approved: bool
    reason: str
    shares: int = 0
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0


class RiskEngine:
    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()
        self.daily_loss_limit = DailyLossLimit(self.config.daily_loss_limit_pct)
        self.drawdown_breaker = MaxDrawdownBreaker(self.config.max_drawdown_pct)
        self.day_trade_tracker = DayTradeTracker(self.config.pdt_equity_threshold)

    def update_equity(self, as_of_date: date, equity: float) -> None:
        """Feed the latest equity mark so the circuit breakers can evaluate it."""
        self.daily_loss_limit.update(as_of_date, equity)
        self.drawdown_breaker.update(equity)

    def evaluate_entry(
        self,
        as_of_date: date,
        entry_price: float,
        equity: float,
        trading_calendar: list[date],
        is_intended_day_trade: bool,
        current_gross_exposure: float = 0.0,
    ) -> TradeDecision:
        """Check and size a proposed entry. Does NOT itself record a day trade —
        callers must call `self.day_trade_tracker.record_day_trade(date)`
        explicitly once they know for certain a round trip closed same-day
        (typically at exit, not entry — a caller often can't know in advance
        whether a position will end up being a day trade). Bundling the
        record into this method previously caused every trade evaluated with
        is_intended_day_trade=True to be recorded here *and* recorded again
        by the caller at exit, double-counting every actual day trade.
        """
        if self.drawdown_breaker.is_tripped:
            return TradeDecision(False, "max_drawdown_circuit_breaker_tripped")
        if self.daily_loss_limit.is_tripped:
            return TradeDecision(False, "daily_loss_limit_tripped")
        if is_intended_day_trade and not self.day_trade_tracker.can_open_day_trade(
            as_of_date, trading_calendar, equity
        ):
            return TradeDecision(False, "pdt_limit_would_be_violated")

        shares = fixed_fractional_shares(
            equity=equity,
            entry_price=entry_price,
            stop_loss_pct=self.config.barrier_config.stop_loss_pct,
            risk_per_trade_pct=self.config.risk_per_trade_pct,
            max_position_fraction=self.config.max_position_fraction,
        )
        if shares <= 0:
            return TradeDecision(False, "position_size_zero")

        # Aggregate check: per-position limits don't compose into a portfolio
        # limit. Trim the order to whatever headroom is left, and refuse
        # outright if there is none.
        headroom = equity * self.config.max_gross_exposure - current_gross_exposure
        if headroom <= 0:
            return TradeDecision(False, "max_gross_exposure_reached")
        affordable_shares = int(headroom // entry_price)
        if affordable_shares <= 0:
            return TradeDecision(False, "max_gross_exposure_reached")
        shares = min(shares, affordable_shares)

        stop_price = entry_price * (1 - self.config.barrier_config.stop_loss_pct)
        target_price = entry_price * (1 + self.config.barrier_config.profit_target_pct)

        return TradeDecision(
            approved=True,
            reason="approved",
            shares=shares,
            stop_loss_price=stop_price,
            take_profit_price=target_price,
        )
