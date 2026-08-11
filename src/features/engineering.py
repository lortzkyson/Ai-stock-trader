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
RSI_WINDOW = 14
RANGE_POSITION_WINDOW = 20
SESSION_START = "09:30"

FEATURE_COLUMNS = (
    [f"ret_{w}" for w in MOMENTUM_WINDOWS]
    + [f"vol_{w}" for w in VOLATILITY_WINDOWS]
    + ["rel_vol_30", "vwap_dev", "minutes_since_open", "rsi_14", "range_position_20"]
)


def _minutes_since_open(
    timestamp_col: pd.Series,
    session_start: str = SESSION_START,
    session_tz: str = "America/New_York",
) -> pd.Series:
    """Minutes elapsed since the session open — a deterministic function of the
    bar's own timestamp (captures open/close intraday effects, no lookahead).

    Real cached bars are stored in UTC (see src/data/quality.py's own
    UTC-vs-exchange-local fix) — converting to `session_tz` first is required,
    not optional, or "session open" ends up meaning midnight-UTC-plus-9:30
    rather than 9:30 ET. Synthetic test fixtures that pre-localize timestamps
    to America/New_York can hide this bug; see the regression test using
    UTC-timestamped data.
    """
    ts = pd.to_datetime(timestamp_col)
    ts = ts.dt.tz_localize(session_tz) if ts.dt.tz is None else ts.dt.tz_convert(session_tz)
    hour, minute = (int(x) for x in session_start.split(":"))
    session_open = ts.dt.normalize() + pd.Timedelta(hours=hour, minutes=minute)
    return (ts - session_open).dt.total_seconds() / 60.0


def _rsi(close: pd.Series, window: int = RSI_WINDOW) -> pd.Series:
    """Relative Strength Index (simple rolling-mean variant, not Wilder's smoothing)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _range_position(
    close: pd.Series, high: pd.Series, low: pd.Series, window: int = RANGE_POSITION_WINDOW
) -> pd.Series:
    """Where price sits in its recent high/low range: 0 = at the low, 1 = at the high."""
    rolling_low = low.rolling(window).min()
    rolling_high = high.rolling(window).max()
    span = (rolling_high - rolling_low).replace(0, np.nan)
    return (close - rolling_low) / span


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

    df["minutes_since_open"] = _minutes_since_open(df["timestamp"])
    df["rsi_14"] = _rsi(close)
    df["range_position_20"] = _range_position(close, df["high"], df["low"])

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
