"""Alpaca's actual fee schedule (confirmed via Alpaca's published fee docs, 2026-08):
$0 commission on US stock/ETF trades. SEC fee and FINRA TAF apply to *sell*
orders only, and are regulatory pass-throughs (not broker markup) — see
docs/data_feed_decision.md-adjacent research notes in commit history.
"""

from __future__ import annotations

SEC_FEE_RATE = 23.10 / 1_000_000  # per dollar of sale proceeds, sells only
FINRA_TAF_PER_SHARE = 0.000119  # sells only
FINRA_TAF_MAX_PER_ORDER = 5.95


def commission(shares: int, price: float) -> float:
    return 0.0  # Alpaca: $0 commission on US stocks/ETFs, both buys and sells


def sell_regulatory_fees(shares: int, price: float) -> float:
    proceeds = shares * price
    sec_fee = proceeds * SEC_FEE_RATE
    taf = min(shares * FINRA_TAF_PER_SHARE, FINRA_TAF_MAX_PER_ORDER)
    return sec_fee + taf
