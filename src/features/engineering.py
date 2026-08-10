"""Lookahead-safe feature engineering for intraday/day-trading signals.

Every feature at time t uses only bar t and earlier bars — never a future
bar. "Using bar t" is not lookahead: by the close of bar t, that bar's own
OHLCV is fully known, and the trading signal it feeds is only acted on
starting the *next* bar (see the fill-on-next-open assumption in Phase 5's
backtester), so this is using currently-completed information, not future
information.

`bars` must be a single symbol's regular-session bars, sorted by timestamp
ascending, with a `timestamp` column (see clean_bars / filter_regular_session
in src/data/quality.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MOMENTUM_WINDOWS = (5, 15, 30, 60)
VOLATILITY_WINDOWS = (30, 60)
RELATIVE_VOLUME_WINDOW = 30

FEATURE_COLUMNS = (
    [f"ret_{w}" for w in MOMENTUM_WINDOWS]
    + [f"vol_{w}" for w in VOLATILITY_WINDOWS]
    + ["rel_vol_30", "vwap_dev"]
)


def add_features(bars: pd.DataFrame) -> pd.DataFrame:
    df = bars.copy()
    close = df["close"]
    one_bar_return = close.pct_change()

    for w in MOMENTUM_WINDOWS:
        df[f"ret_{w}"] = close.pct_change(w)

    for w in VOLATILITY_WINDOWS:
        df[f"vol_{w}"] = one_bar_return.rolling(w).std()

    df["rel_vol_30"] = df["volume"] / df["volume"].rolling(RELATIVE_VOLUME_WINDOW).mean()

    session_date = pd.to_datetime(df["timestamp"]).dt.date
    dollar_volume = close * df["volume"]
    cum_dollar_volume = dollar_volume.groupby(session_date).cumsum()
    cum_volume = df["volume"].groupby(session_date).cumsum()
    session_vwap = cum_dollar_volume / cum_volume
    df["vwap_dev"] = (close - session_vwap) / session_vwap

    return df


def assert_no_lookahead(bars: pd.DataFrame, cutoff_frac: float = 0.8) -> None:
    """Raise if any feature at/before `cutoff_frac` changes when future rows are altered.

    Mutates every row after the cutoff to extreme, clearly-out-of-distribution
    values and recomputes features; a lookahead-safe feature set must produce
    byte-identical values up to the cutoff either way.
    """
    n = len(bars)
    cutoff = int(n * cutoff_frac)

    original = add_features(bars)

    mutated_input = bars.copy()
    mutated_input["volume"] = mutated_input["volume"].astype(float)
    future = mutated_input.index[cutoff:]
    rng = np.random.default_rng(seed=0)
    mutated_input.loc[future, "close"] = rng.uniform(1e6, 2e6, size=len(future))
    mutated_input.loc[future, "high"] = mutated_input.loc[future, "close"] * 1.5
    mutated_input.loc[future, "low"] = mutated_input.loc[future, "close"] * 0.5
    mutated_input.loc[future, "open"] = mutated_input.loc[future, "close"]
    mutated_input.loc[future, "volume"] = rng.uniform(1e9, 2e9, size=len(future))

    mutated = add_features(mutated_input)

    for col in FEATURE_COLUMNS:
        pd.testing.assert_series_equal(
            original[col].iloc[:cutoff],
            mutated[col].iloc[:cutoff],
            check_names=False,
        )
