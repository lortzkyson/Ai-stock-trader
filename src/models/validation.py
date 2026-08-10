"""Walk-forward (rolling-origin) validation with purge/embargo.

Random k-fold cross-validation leaks future information into training for
time series (docs/pre-mortem.md guard #8): a randomly-selected validation
row can sit chronologically before training rows the model was fit on.
Every split here is strictly time-ordered instead — train always precedes
test.

Purge/embargo: triple-barrier labels look forward up to
`TripleBarrierConfig.max_holding_bars` (~3 trading days by default), so a
training row near the train/test boundary can have a label whose outcome
was determined using price action that actually falls inside the test
period — a leak. `embargo_days` drops that many trading days off the end of
each fold's training window to remove the overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    train_start: date
    train_end: date  # already purged/embargoed
    test_start: date
    test_end: date


def generate_walk_forward_folds(
    trading_dates: list[date],
    n_folds: int = 5,
    embargo_days: int = 4,
) -> list[WalkForwardFold]:
    n = len(trading_dates)
    block_size = n // (n_folds + 1)
    if block_size < embargo_days + 1:
        raise ValueError("not enough trading dates to fit n_folds with this embargo")

    folds = []
    for k in range(n_folds):
        train_end_idx = block_size * (k + 1) - 1
        test_start_idx = train_end_idx + 1
        test_end_idx = block_size * (k + 2) - 1 if k < n_folds - 1 else n - 1
        if test_start_idx > test_end_idx:
            break

        purged_train_end_idx = max(0, train_end_idx - embargo_days)
        folds.append(
            WalkForwardFold(
                fold_id=k,
                train_start=trading_dates[0],
                train_end=trading_dates[purged_train_end_idx],
                test_start=trading_dates[test_start_idx],
                test_end=trading_dates[test_end_idx],
            )
        )
    return folds


def split_fold(
    df: pd.DataFrame, fold: WalkForwardFold, date_col: str = "date"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = df[date_col]
    train = df.loc[(dates >= fold.train_start) & (dates <= fold.train_end)]
    test = df.loc[(dates >= fold.test_start) & (dates <= fold.test_end)]
    return train, test
