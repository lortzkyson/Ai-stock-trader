from __future__ import annotations

import pytest

from risk.position_sizing import capped_kelly_fraction, fixed_fractional_shares


def test_fixed_fractional_shares_risk_based() -> None:
    # Risk 1% of $100,000 = $1,000. Stop distance = $100 * 1% = $1/share.
    # -> 1000 shares by risk. max_position_fraction=1.0 means the notional
    # cap (all $100,000 / $100 = 1000 shares) doesn't bind first.
    shares = fixed_fractional_shares(
        equity=100_000,
        entry_price=100.0,
        stop_loss_pct=0.01,
        risk_per_trade_pct=0.01,
        max_position_fraction=1.0,
    )
    assert shares == 1000


def test_fixed_fractional_shares_notional_cap_binds() -> None:
    # A very tight stop would imply a huge share count by risk alone; the
    # notional cap should bind instead.
    shares = fixed_fractional_shares(
        equity=100_000,
        entry_price=100.0,
        stop_loss_pct=0.0001,
        risk_per_trade_pct=0.01,
        max_position_fraction=0.10,  # cap notional at $10,000 -> 100 shares
    )
    assert shares == 100


def test_fixed_fractional_shares_invalid_inputs_return_zero() -> None:
    assert fixed_fractional_shares(0, 100, 0.01, 0.01, 0.5) == 0
    assert fixed_fractional_shares(100_000, 0, 0.01, 0.01, 0.5) == 0
    assert fixed_fractional_shares(100_000, 100, 0, 0.01, 0.5) == 0


def test_capped_kelly_fraction_positive_edge() -> None:
    # win_rate=0.4, reward:risk=2 -> full kelly = 0.4 - 0.6/2 = 0.1
    # half-kelly = 0.05, under a 0.05 cap -> exactly 0.05
    frac = capped_kelly_fraction(
        win_rate=0.4, reward_risk_ratio=2.0, kelly_cap=0.5, max_fraction=0.5
    )
    assert frac == pytest.approx(0.05)


def test_capped_kelly_fraction_negative_edge_is_zero() -> None:
    frac = capped_kelly_fraction(win_rate=0.2, reward_risk_ratio=1.0)
    assert frac == 0.0


def test_capped_kelly_fraction_hits_hard_cap() -> None:
    # Very favorable inputs would suggest oversizing; max_fraction must bind.
    frac = capped_kelly_fraction(
        win_rate=0.9, reward_risk_ratio=5.0, kelly_cap=1.0, max_fraction=0.05
    )
    assert frac == pytest.approx(0.05)
