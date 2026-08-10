from __future__ import annotations

import pytest

from backtest.costs import commission, sell_regulatory_fees


def test_commission_is_zero() -> None:
    assert commission(100, 50.0) == 0.0


def test_sell_regulatory_fees_typical_trade() -> None:
    # 100 shares @ $50 = $5,000 proceeds.
    # SEC: 5000 * 23.10/1_000_000 = 0.1155
    # TAF: 100 * 0.000119 = 0.0119 (well under the $5.95 cap)
    fees = sell_regulatory_fees(100, 50.0)
    assert fees == pytest.approx(0.1155 + 0.0119, abs=1e-6)


def test_sell_regulatory_fees_taf_caps_at_order_max() -> None:
    # Huge share count should cap the TAF portion at $5.95, not scale unbounded.
    fees_huge = sell_regulatory_fees(1_000_000, 1.0)
    sec_fee = 1_000_000 * 1.0 * (23.10 / 1_000_000)
    assert fees_huge == pytest.approx(sec_fee + 5.95, abs=1e-6)
