from __future__ import annotations

import pandas as pd
import pytest

from backtest.fills import FillConfig, apply_exit_slippage, simulate_entry_fill


def _bar(open_: float, high: float, low: float, close: float) -> pd.Series:
    return pd.Series({"open": open_, "high": high, "low": low, "close": close})


def test_market_order_fills_at_open_plus_slippage() -> None:
    bar = _bar(100.0, 101.0, 99.0, 100.5)
    config = FillConfig(order_type="market", slippage_bps=10.0)

    fill = simulate_entry_fill(bar, config)

    assert fill == pytest.approx(100.0 * 1.001)


def test_limit_order_fills_when_price_crosses_it() -> None:
    bar = _bar(open_=100.0, high=101.0, low=98.5, close=99.0)
    config = FillConfig(order_type="limit", limit_offset_bps=100.0)  # limit = 100 * 0.99 = 99.0

    fill = simulate_entry_fill(bar, config)

    assert fill == pytest.approx(99.0)


def test_limit_order_does_not_fill_when_price_never_crosses() -> None:
    bar = _bar(open_=100.0, high=101.0, low=99.5, close=100.5)
    config = FillConfig(order_type="limit", limit_offset_bps=100.0)  # limit=99.0, low only hit 99.5

    fill = simulate_entry_fill(bar, config)

    assert fill is None


def test_limit_order_fills_at_open_if_it_gaps_through_the_limit() -> None:
    bar = _bar(open_=95.0, high=96.0, low=94.0, close=95.5)  # opens below the intended limit
    config = FillConfig(order_type="limit", limit_offset_bps=100.0)  # limit would be 94.05

    fill = simulate_entry_fill(bar, config)

    assert fill == pytest.approx(94.05)


def test_apply_exit_slippage_reduces_sell_price() -> None:
    assert apply_exit_slippage(100.0, slippage_bps=10.0) == pytest.approx(99.9)
