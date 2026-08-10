from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.labeling import TripleBarrierConfig, label_triple_barrier, report_class_balance


def make_bars(closes: list[float], highs: list[float], lows: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"close": closes, "high": highs, "low": lows})


def test_profit_target_hit_first() -> None:
    closes = [100, 100.5, 101, 100.8, 100.9] + [100] * 20
    highs = [100.5, 101, 102.5, 101, 101] + [100.2] * 20
    lows = [99.8, 100, 100.5, 100.5, 100.5] + [99.9] * 20
    bars = make_bars(closes, highs, lows)

    labeled = label_triple_barrier(
        bars, TripleBarrierConfig(profit_target_pct=0.02, stop_loss_pct=0.01, max_holding_bars=10)
    )

    assert labeled.loc[0, "label"] == 1
    assert labeled.loc[0, "exit_bar_offset"] == 2
    assert labeled.loc[0, "realized_return"] == pytest.approx(0.02)


def test_stop_loss_hit_first() -> None:
    closes = [100, 99.8, 99.5, 98.5, 100] + [100] * 20
    highs = [100.2, 100, 99.8, 99, 100.5] + [100.2] * 20
    lows = [99.5, 99.2, 98.9, 98.5, 99.8] + [99.9] * 20
    bars = make_bars(closes, highs, lows)

    labeled = label_triple_barrier(
        bars, TripleBarrierConfig(profit_target_pct=0.02, stop_loss_pct=0.01, max_holding_bars=10)
    )

    assert labeled.loc[0, "label"] == -1
    assert labeled.loc[0, "exit_bar_offset"] == 2
    assert labeled.loc[0, "realized_return"] == pytest.approx(-0.01)


def test_tie_within_same_bar_stop_wins() -> None:
    closes = [100] + [100] * 20
    highs = [100.2, 103] + [100.2] * 19
    lows = [99.5, 98] + [99.5] * 19
    bars = make_bars(closes, highs, lows)

    labeled = label_triple_barrier(
        bars, TripleBarrierConfig(profit_target_pct=0.02, stop_loss_pct=0.01, max_holding_bars=10)
    )

    assert labeled.loc[0, "label"] == -1
    assert labeled.loc[0, "exit_bar_offset"] == 1


def test_timeout_when_no_barrier_hit() -> None:
    n = 15
    closes = [100 + 0.01 * i for i in range(n)]
    highs = [c + 0.05 for c in closes]
    lows = [c - 0.05 for c in closes]
    bars = make_bars(closes, highs, lows)

    labeled = label_triple_barrier(
        bars, TripleBarrierConfig(profit_target_pct=0.02, stop_loss_pct=0.01, max_holding_bars=10)
    )

    assert labeled.loc[0, "label"] == 0
    assert labeled.loc[0, "exit_bar_offset"] == 10
    expected_return = closes[10] / closes[0] - 1
    assert labeled.loc[0, "realized_return"] == pytest.approx(expected_return)


def test_insufficient_horizon_is_nan() -> None:
    n = 8
    closes = [100] * n
    highs = [100.1] * n
    lows = [99.9] * n
    bars = make_bars(closes, highs, lows)

    labeled = label_triple_barrier(bars, TripleBarrierConfig(max_holding_bars=10))

    assert labeled["label"].isna().all()


def test_report_class_balance_flags_skew() -> None:
    labels = pd.Series([1] * 90 + [-1] * 5 + [0] * 5 + [np.nan] * 10)

    report = report_class_balance(labels, skew_threshold=0.10)

    assert report["n_valid"] == 100
    assert report["n_dropped_insufficient_horizon"] == 10
    assert report["is_skewed"] is True


def test_report_class_balance_not_skewed_when_balanced() -> None:
    labels = pd.Series([1] * 40 + [-1] * 30 + [0] * 30)

    report = report_class_balance(labels, skew_threshold=0.10)

    assert report["is_skewed"] is False
