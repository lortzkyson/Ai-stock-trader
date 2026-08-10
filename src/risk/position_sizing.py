"""Position sizing.

Fixed-fractional is the default: risk a fixed % of equity per trade based on
stop-loss distance. It's simple and doesn't require a reliable win-rate/payoff
estimate the way Kelly does. Capped (half-)Kelly is offered as an alternative
but isn't the default — Kelly sizing is exactly most sensitive to estimation
error in win-rate/payoff, and docs/model_card.md's own finding is that this
model doesn't yet show a clear edge over random entries. Sizing aggressively
off an edge that hasn't been demonstrated would compound that problem.
"""

from __future__ import annotations


def fixed_fractional_shares(
    equity: float,
    entry_price: float,
    stop_loss_pct: float,
    risk_per_trade_pct: float,
    max_position_fraction: float,
) -> int:
    """Size a long position so a stop-loss hit costs `risk_per_trade_pct` of equity.

    Also caps notional exposure at `max_position_fraction` of equity
    regardless of stop distance, so an unusually tight stop can't imply an
    oversized position.
    """
    if entry_price <= 0 or stop_loss_pct <= 0 or equity <= 0:
        return 0

    risk_dollars = equity * risk_per_trade_pct
    stop_distance = entry_price * stop_loss_pct
    shares_by_risk = risk_dollars / stop_distance

    max_notional = equity * max_position_fraction
    shares_by_notional_cap = max_notional / entry_price

    return int(min(shares_by_risk, shares_by_notional_cap))


def capped_kelly_fraction(
    win_rate: float,
    reward_risk_ratio: float,
    kelly_cap: float = 0.5,
    max_fraction: float = 0.05,
) -> float:
    """Half-Kelly (by default) fraction of equity, hard-capped at `max_fraction`.

    Full Kelly (`kelly_cap=1.0`) assumes exact knowledge of win_rate and
    reward_risk_ratio; realistic estimation error gets amplified by full
    Kelly's aggressive sizing. Halving it (the standard practice this
    defaults to) and adding a hard ceiling on top buffers that.
    """
    if reward_risk_ratio <= 0:
        return 0.0
    full_kelly = win_rate - (1 - win_rate) / reward_risk_ratio
    full_kelly = max(full_kelly, 0.0)  # never size a position on a negative edge
    return min(full_kelly * kelly_cap, max_fraction)
