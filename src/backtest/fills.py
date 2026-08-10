"""Fill simulation for entries.

Two order types:
- market: always fills, at the scheduled bar's open plus slippage in the
  adverse direction (never at a better price than what was actually quoted).
- limit: only fills if the scheduled bar's range actually crosses the limit
  price — otherwise the signal goes unfilled, same as a real limit order
  that never gets touched. This is what "including the possibility of a
  limit order not filling" means in practice.

Exits (stop-loss / take-profit / max-holding) are modeled as market fills —
a triggered stop becomes a market order in practice, and take-profit exits
here fill at exactly the barrier level (never better), which is the same
conservative assumption src/features/labeling.py uses. A limit-order
take-profit that could itself go unfilled is a possible future extension,
not implemented here to keep scope contained.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FillConfig:
    order_type: str = "market"  # "market" or "limit"
    slippage_bps: float = 5.0
    limit_offset_bps: float = 0.0  # limit orders only: buy limit = open * (1 - offset)
    latency_bars: int = 0  # additional bars of delay beyond the mandatory next-bar-open


def simulate_entry_fill(bar: pd.Series, config: FillConfig) -> float | None:
    """Return the long-entry fill price on this bar, or None if unfilled (limit orders only)."""
    scheduled_open = float(bar["open"])

    if config.order_type == "market":
        return scheduled_open * (1 + config.slippage_bps / 10_000)

    if config.order_type == "limit":
        limit_price = scheduled_open * (1 - config.limit_offset_bps / 10_000)
        if float(bar["low"]) <= limit_price:
            # Can't fill better than the limit; if price gaps down through
            # it, the fill is at the (better, for us) open instead.
            return min(limit_price, scheduled_open)
        return None

    raise ValueError(f"unknown order_type: {config.order_type}")


def apply_exit_slippage(barrier_price: float, slippage_bps: float) -> float:
    """Sell fills slip down from the barrier level (adverse direction for a long)."""
    return barrier_price * (1 - slippage_bps / 10_000)
