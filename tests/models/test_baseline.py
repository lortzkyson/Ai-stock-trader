from __future__ import annotations

import pandas as pd
import pytest

from models.baseline import buy_and_hold_baseline, random_entry_baseline


def test_random_entry_baseline_samples_requested_trade_count() -> None:
    test_df = pd.DataFrame(
        {
            "realized_return": [0.01, -0.01, 0.02, -0.02, 0.0, 0.01, -0.01],
            "label": [1, -1, 1, -1, 0, 1, -1],
            "date": ["2026-01-05"] * 7,
        }
    )

    result = random_entry_baseline(test_df, n_trades=3, seed=1)

    assert result["n_trades"] == 3


def test_random_entry_baseline_empty_when_zero_trades() -> None:
    test_df = pd.DataFrame({"realized_return": [0.01], "label": [1], "date": ["2026-01-05"]})
    result = random_entry_baseline(test_df, n_trades=0)
    assert result["n_trades"] == 0


def test_buy_and_hold_baseline_averages_symbol_returns() -> None:
    bars_by_symbol = {
        "AAA": pd.DataFrame({"close": [100.0, 110.0]}),  # +10%
        "BBB": pd.DataFrame({"close": [50.0, 45.0]}),  # -10%
    }
    result = buy_and_hold_baseline(bars_by_symbol)
    assert result == pytest.approx(0.0, abs=1e-9)


def test_buy_and_hold_baseline_nan_when_no_data() -> None:
    import math

    assert math.isnan(buy_and_hold_baseline({}))
