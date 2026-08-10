from __future__ import annotations

import pandas as pd
import pytest

from data.universe import LiquidityFilterConfig, filter_by_liquidity, load_seed_universe


def test_load_seed_universe_returns_symbols() -> None:
    symbols = load_seed_universe()
    assert "AAPL" in symbols
    assert len(symbols) > 0


def test_filter_by_liquidity_excludes_low_volume_and_penny_stocks() -> None:
    daily_bars = pd.DataFrame(
        [
            # LIQUID: high price, high volume -> keep
            {"symbol": "LIQUID", "close": 100.0, "volume": 1_000_000},
            {"symbol": "LIQUID", "close": 101.0, "volume": 1_000_000},
            # THIN: passes price range but volume too low -> drop
            {"symbol": "THIN", "close": 50.0, "volume": 100},
            {"symbol": "THIN", "close": 50.0, "volume": 100},
            # PENNY: high volume but price below floor -> drop
            {"symbol": "PENNY", "close": 0.50, "volume": 50_000_000},
            {"symbol": "PENNY", "close": 0.50, "volume": 50_000_000},
        ]
    )

    config = LiquidityFilterConfig(min_avg_dollar_volume=5_000_000, min_price=5.0, max_price=1000.0)
    kept = filter_by_liquidity(daily_bars, config=config)

    assert kept == ["LIQUID"]


def test_filter_by_liquidity_requires_expected_columns() -> None:
    bad = pd.DataFrame({"symbol": ["A"], "close": [1.0]})
    with pytest.raises(ValueError):
        filter_by_liquidity(bad)
