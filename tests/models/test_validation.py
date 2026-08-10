from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from models.validation import generate_walk_forward_folds, split_fold


def _dates(n: int) -> list[date]:
    start = date(2026, 1, 1)
    return [start + timedelta(days=i) for i in range(n)]


def test_folds_are_strictly_time_ordered() -> None:
    dates = _dates(60)
    folds = generate_walk_forward_folds(dates, n_folds=5, embargo_days=2)

    assert len(folds) == 5
    for fold in folds:
        assert fold.train_end < fold.test_start
        assert fold.test_start <= fold.test_end


def test_embargo_removes_dates_from_train_end() -> None:
    dates = _dates(60)
    no_embargo = generate_walk_forward_folds(dates, n_folds=5, embargo_days=0)
    with_embargo = generate_walk_forward_folds(dates, n_folds=5, embargo_days=3)

    # Embargoed train_end for each fold should be strictly earlier.
    for a, b in zip(no_embargo, with_embargo):
        assert b.train_end < a.train_end


def test_raises_when_not_enough_dates_for_embargo() -> None:
    dates = _dates(5)
    with pytest.raises(ValueError):
        generate_walk_forward_folds(dates, n_folds=5, embargo_days=4)


def test_split_fold_respects_boundaries() -> None:
    dates = _dates(60)
    folds = generate_walk_forward_folds(dates, n_folds=5, embargo_days=2)
    fold = folds[0]

    df = pd.DataFrame({"date": dates, "value": range(len(dates))})
    train, test = split_fold(df, fold)

    assert (train["date"] <= fold.train_end).all()
    assert (train["date"] >= fold.train_start).all()
    assert (test["date"] >= fold.test_start).all()
    assert (test["date"] <= fold.test_end).all()
    assert len(train) + len(test) < len(df)  # embargoed dates belong to neither
