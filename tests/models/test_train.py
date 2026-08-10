from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from data.cache import write_cache
from features.labeling import TripleBarrierConfig
from models.dataset import build_dataset
from models.train import aggregate_oos_metrics, regime_split_metrics, run_walk_forward

from .conftest import make_random_walk_bars

TRADING_DAYS = [
    "2026-01-05",
    "2026-01-06",
    "2026-01-07",
    "2026-01-08",
    "2026-01-09",
    "2026-01-12",
    "2026-01-13",
    "2026-01-14",
    "2026-01-15",
    "2026-01-16",
    "2026-01-20",
    "2026-01-21",
    "2026-01-22",
    "2026-01-23",
    "2026-01-26",
    "2026-01-27",
    "2026-01-28",
    "2026-01-29",
    "2026-01-30",
    "2026-02-02",
    "2026-02-03",
    "2026-02-04",
    "2026-02-05",
    "2026-02-06",
]


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> pd.DataFrame:
    bars = make_random_walk_bars(TRADING_DAYS, seed=7, bar_std=0.3)
    write_cache(bars, "AAA", "1Min", date(2026, 1, 5), date(2026, 2, 6), cache_dir=tmp_path)

    return build_dataset(
        ["AAA"],
        date(2026, 1, 5),
        date(2026, 2, 6),
        barrier_config=TripleBarrierConfig(
            profit_target_pct=0.01, stop_loss_pct=0.01, max_holding_bars=15
        ),
        cache_dir=tmp_path,
    )


def test_run_walk_forward_produces_fold_metrics_and_oos(synthetic_dataset: pd.DataFrame) -> None:
    fold_metrics, oos = run_walk_forward(synthetic_dataset, n_folds=3, embargo_days=1)

    assert len(fold_metrics) > 0
    for fm in fold_metrics:
        assert fm.n_train > 0
        assert fm.n_test > 0
        assert 0.0 <= fm.accuracy <= 1.0
    assert not oos.empty
    assert {"predicted", "proba", "fold_id"}.issubset(oos.columns)


def test_folds_never_train_on_future_data(synthetic_dataset: pd.DataFrame) -> None:
    fold_metrics, _ = run_walk_forward(synthetic_dataset, n_folds=3, embargo_days=1)
    for fm in fold_metrics:
        assert fm.train_end < fm.test_start


def test_aggregate_and_regime_metrics_run_without_error(synthetic_dataset: pd.DataFrame) -> None:
    _, oos = run_walk_forward(synthetic_dataset, n_folds=3, embargo_days=1)

    agg = aggregate_oos_metrics(oos)
    assert "expectancy" in agg

    regimes = regime_split_metrics(synthetic_dataset, oos)
    assert isinstance(regimes, dict)
