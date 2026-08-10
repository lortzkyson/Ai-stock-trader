"""Live signal generation.

Reuses the exact same feature-computation code as the backtester
(`features.engineering.add_features`) rather than a separate live
reimplementation — a training/serving mismatch here is a common, hard-to-notice
source of live underperformance. See tests/execution/test_parity.py for the
check that feeding identical bars through both paths gives identical features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from features.engineering import FEATURE_COLUMNS, add_features

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = REPO_ROOT / "data" / "models" / "production_model.joblib"


def load_production_model(path: Path = DEFAULT_MODEL_PATH) -> Any:
    return joblib.load(path)


def compute_live_signal(
    bars: pd.DataFrame, model: Any, probability_threshold: float = 0.5
) -> tuple[int, float]:
    """bars: one symbol's regular-session bars, sorted ascending, most recent bar last.

    Returns (predicted, probability). predicted=0/proba=nan if there isn't
    enough warmup history yet for the rolling-window features to be valid.
    """
    featured = add_features(bars)
    latest = featured.iloc[[-1]]
    if latest[FEATURE_COLUMNS].isna().any(axis=1).iloc[0]:
        return 0, float("nan")

    proba = float(model.predict_proba(latest[FEATURE_COLUMNS])[0, 1])
    predicted = int(proba >= probability_threshold)
    return predicted, proba
