"""Final holdout window (see docs/pre-mortem.md §5). Pinned once, in Phase 2.

No code in Phases 3-7 (features, training, backtesting) should read data
from this window — it's used exactly once, in Phase 8, as the last check
before considering real-money deployment.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

HOLDOUT_START = date(2026, 5, 1)
HOLDOUT_END = date(2026, 7, 31)


def exclude_holdout(bars: pd.DataFrame, exclude: bool = True) -> pd.DataFrame:
    """Drop rows falling inside the final holdout window.

    `exclude` defaults to True so holdout data has to be opted into
    deliberately (Phase 8 only), never left in by omission.
    """
    if not exclude:
        return bars
    ts = pd.to_datetime(bars["timestamp"])
    in_holdout = (ts.dt.date >= HOLDOUT_START) & (ts.dt.date <= HOLDOUT_END)
    return bars.loc[~in_holdout].reset_index(drop=True)
