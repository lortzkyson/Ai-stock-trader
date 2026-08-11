"""Walk-forward model training. See docs/pre-mortem.md guards #3, #4, #8.

Trains a gradient-boosted-trees binary classifier per fold — kept
deliberately simple before reaching for anything deep-learning-based, since
nothing here suggests the extra complexity is justified yet — predicting
P(profit target hit first) for a long entry, validated strictly out-of-time
per fold (never touching the final holdout from docs/pre-mortem.md §5).

Uses scikit-learn's HistGradientBoostingClassifier rather than LightGBM/XGBoost:
this machine has no Homebrew and no `libomp`, which both of those need for
their compiled backend, and there's no way to install it here. HGB is the
same family of model (gradient-boosted trees) with no external native
dependency, so nothing about the approach changes — just the library.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from models.dataset import ALL_FEATURE_COLUMNS as FEATURE_COLUMNS
from models.metrics import compute_signal_metrics
from models.validation import generate_walk_forward_folds, split_fold


@dataclass
class FoldMetrics:
    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_test: int
    accuracy: float
    n_trades: int
    win_rate: float
    expectancy: float
    sharpe: float
    max_drawdown: float


def train_fold_model(train_df: pd.DataFrame) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=5,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(train_df[FEATURE_COLUMNS], train_df["target"])
    return model


def run_walk_forward(
    dataset: pd.DataFrame,
    n_folds: int = 5,
    embargo_days: int = 4,
    probability_threshold: float = 0.5,
) -> tuple[list[FoldMetrics], pd.DataFrame]:
    trading_dates: list[date] = sorted(dataset["date"].unique())
    folds = generate_walk_forward_folds(trading_dates, n_folds=n_folds, embargo_days=embargo_days)

    fold_metrics: list[FoldMetrics] = []
    oos_frames: list[pd.DataFrame] = []

    for fold in folds:
        train_df, test_df = split_fold(dataset, fold)
        if len(train_df) == 0 or len(test_df) == 0:
            continue

        model = train_fold_model(train_df)
        proba = model.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]
        predicted = (proba >= probability_threshold).astype(int)
        accuracy = float((predicted == test_df["target"].to_numpy()).mean())

        traded = test_df.loc[predicted == 1]
        signal_metrics = compute_signal_metrics(
            traded["realized_return"].to_numpy(),
            traded["label"].to_numpy(),
            traded["date"].to_numpy(),
        )

        fold_metrics.append(
            FoldMetrics(
                fold_id=fold.fold_id,
                train_start=str(fold.train_start),
                train_end=str(fold.train_end),
                test_start=str(fold.test_start),
                test_end=str(fold.test_end),
                n_train=len(train_df),
                n_test=len(test_df),
                accuracy=accuracy,
                **signal_metrics,
            )
        )

        oos = test_df[["timestamp", "symbol", "date", "label", "realized_return"]].copy()
        oos["fold_id"] = fold.fold_id
        oos["predicted"] = predicted
        oos["proba"] = proba
        oos_frames.append(oos)

    oos_predictions = pd.concat(oos_frames, ignore_index=True) if oos_frames else pd.DataFrame()
    return fold_metrics, oos_predictions


def regime_split_metrics(dataset: pd.DataFrame, oos_predictions: pd.DataFrame) -> dict:
    """Split out-of-sample trades by a realized-volatility regime (median split by day)."""
    if oos_predictions.empty:
        return {}

    daily_vol = dataset.groupby("date")["vol_30"].mean()
    median_vol = daily_vol.median()
    high_vol_days = set(daily_vol[daily_vol >= median_vol].index)

    traded = oos_predictions.loc[oos_predictions["predicted"] == 1].copy()
    traded["regime"] = traded["date"].apply(
        lambda d: "high_vol" if d in high_vol_days else "low_vol"
    )

    out = {}
    for regime, group in traded.groupby("regime"):
        out[regime] = compute_signal_metrics(
            group["realized_return"].to_numpy(),
            group["label"].to_numpy(),
            group["date"].to_numpy(),
        )
    return out


def aggregate_oos_metrics(oos_predictions: pd.DataFrame) -> dict:
    if oos_predictions.empty:
        return compute_signal_metrics(np.array([]), np.array([]), np.array([]))
    traded = oos_predictions.loc[oos_predictions["predicted"] == 1]
    return compute_signal_metrics(
        traded["realized_return"].to_numpy(),
        traded["label"].to_numpy(),
        traded["date"].to_numpy(),
    )
