from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from data.cache import write_cache
from features.labeling import TripleBarrierConfig
from models.dataset import build_dataset, build_symbol_dataset

from .conftest import make_random_walk_bars


def test_build_symbol_dataset_raises_when_not_cached(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_symbol_dataset("NOPE", date(2026, 1, 5), date(2026, 1, 5), cache_dir=tmp_path)


def test_build_dataset_combines_symbols(tmp_path: Path) -> None:
    for symbol, seed in [("AAA", 1), ("BBB", 2)]:
        bars = make_random_walk_bars(["2026-01-05"], seed=seed)
        write_cache(bars, symbol, "1Min", date(2026, 1, 5), date(2026, 1, 5), cache_dir=tmp_path)

    dataset = build_dataset(
        ["AAA", "BBB"],
        date(2026, 1, 5),
        date(2026, 1, 5),
        barrier_config=TripleBarrierConfig(
            profit_target_pct=0.01, stop_loss_pct=0.01, max_holding_bars=5
        ),
        cache_dir=tmp_path,
    )

    assert set(dataset["symbol"]) == {"AAA", "BBB"}
    assert "target" in dataset.columns
    assert dataset["target"].isin([0, 1]).all()
    assert dataset["label"].notna().all()
    assert (dataset["timestamp"].diff().dropna() >= pd.Timedelta(0)).all()
