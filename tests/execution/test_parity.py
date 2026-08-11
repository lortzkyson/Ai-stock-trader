"""Backtest/live feature parity — see docs/pre-mortem.md Phase 7.

Feeds identical historical bars through the backtest path (features computed
once, in a batch, over the whole DataFrame — what src/models/dataset.py does)
and the live path (features recomputed on a growing window at each tick, the
way src/execution/signal.py's compute_live_signal actually gets called live)
and asserts identical feature values. This exercises a genuine potential
divergence — batch vs. incremental computation over the same rolling-window
logic — not just the same function trivially compared against itself.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from execution.signal import compute_live_signal
from features.engineering import add_features
from features.ict_concepts import add_ict_features
from models.dataset import ALL_FEATURE_COLUMNS as FEATURE_COLUMNS

from .conftest import make_clean_bars


def _batch_features(bars: pd.DataFrame) -> pd.DataFrame:
    return add_ict_features(add_features(bars))


class _DummyModel:
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.tile([0.4, 0.6], (len(X), 1))


def test_live_and_backtest_feature_paths_agree_on_identical_bars() -> None:
    bars = make_clean_bars(["2026-01-05", "2026-01-06"])
    warmup = 60  # enough bars for the longest rolling window (60) to be valid

    backtest_features = _batch_features(bars)
    model = _DummyModel()

    for i in range(warmup, len(bars)):
        window = bars.iloc[: i + 1]

        predicted, proba = compute_live_signal(window, model)
        assert not math.isnan(proba)  # enough warmup here for a real prediction

        live_row = _batch_features(window).iloc[-1]
        backtest_row = backtest_features.iloc[i]
        for col in FEATURE_COLUMNS:
            live_val, backtest_val = live_row[col], backtest_row[col]
            if pd.isna(live_val) or pd.isna(backtest_val):
                assert pd.isna(live_val) and pd.isna(backtest_val), col
            else:
                assert live_val == pytest.approx(backtest_val), col


def test_compute_live_signal_returns_no_prediction_before_warmup() -> None:
    bars = make_clean_bars(["2026-01-05"]).iloc[:10]  # far fewer than the 60-bar warmup
    predicted, proba = compute_live_signal(bars, _DummyModel())
    assert predicted == 0
    assert math.isnan(proba)
