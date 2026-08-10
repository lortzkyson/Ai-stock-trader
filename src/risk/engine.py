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
    ) -> TradeDecision:
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

        stop_price = entry_price * (1 - self.config.barrier_config.stop_loss_pct)
        target_price = entry_price * (1 + self.config.barrier_config.profit_target_pct)

        if is_intended_day_trade:
            self.day_trade_tracker.record_day_trade(as_of_date)

        return TradeDecision(
            approved=True,
            reason="approved",
            shares=shares,
            stop_loss_price=stop_price,
            take_profit_price=target_price,
        )
