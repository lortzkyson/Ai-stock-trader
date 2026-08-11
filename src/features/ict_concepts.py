"""ICT-style features: PO3 (Power of Three / Accumulation-Manipulation-Distribution)
and Fair Value Gaps, including Inverse FVG (a gap that gets decisively broken
and flips from support to resistance or vice versa).

Both concepts are operationalized as concrete, computable, lookahead-safe
rules — not a vague "the model learns ICT" black box:

PO3 / AMD: the first `ACCUMULATION_WINDOW_MINUTES` of each session define an
"accumulation" range. A "manipulation" is a sweep — price pokes beyond that
range (grabbing the liquidity resting there) and then closes back inside it,
the classic stop-hunt signature. The expected "distribution" move runs
opposite the sweep direction: a sweep of the range low (downside manipulation)
implies bullish distribution; a sweep of the range high implies bearish
distribution. State (the range, and time since the last sweep) resets every
session — each day's AMD cycle is treated as independent.

Fair Value Gap (FVG): a 3-candle imbalance. A bullish FVG forms when
candle[i-2]'s high is below candle[i]'s low (candle[i-1] never traded in
between) — the gap is expected to act as support if price returns to it. A
bearish FVG is the mirror. Only the single most-recent gap of each type is
tracked (a documented simplification — not a full gap history) as a state
machine: it's "filled" once price trades back through it, or "inverted" if
price *closes* decisively through it (a wick isn't enough) — at that point
it flips polarity: a bullish FVG that gets closed-through becomes resistance
(an Inverse FVG), and vice versa.

Long-only scope: this system doesn't short (see docs/pre-mortem.md). Bearish
signals here (upside sweeps, bearish IFVGs) are still computed and exposed
as features — the model can use "this looks like a short setup" as a reason
to avoid a long entry, which is legitimate signal even without a short book.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ACCUMULATION_WINDOW_MINUTES = 30
SWEEP_LOOKBACK_BARS = 30
RECENCY_CAP_BARS = 390  # one regular session; "no recent event" reads as "a full day ago"

PO3_FEATURE_COLUMNS = [
    "dist_to_accum_high",
    "dist_to_accum_low",
    "swept_accum_high_recent",
    "swept_accum_low_recent",
    "bars_since_sweep",
    "sweep_direction",
]

FVG_FEATURE_COLUMNS = [
    "bull_fvg_active",
    "bull_fvg_dist",
    "bear_fvg_active",
    "bear_fvg_dist",
    "bars_since_bull_ifvg",
    "bars_since_bear_ifvg",
]

ICT_FEATURE_COLUMNS = PO3_FEATURE_COLUMNS + FVG_FEATURE_COLUMNS


def _minutes_since_open(
    timestamp_col: pd.Series, session_start: str = "09:30", session_tz: str = "America/New_York"
) -> pd.Series:
    """See features.engineering._minutes_since_open's docstring: real cached
    bars are UTC, so converting to `session_tz` first is required, not
    optional — this is the same UTC-vs-exchange-local bug class as
    src/data/quality.py's fix, caught here via a real-data regression test.
    """
    ts = pd.to_datetime(timestamp_col)
    ts = ts.dt.tz_localize(session_tz) if ts.dt.tz is None else ts.dt.tz_convert(session_tz)
    hour, minute = (int(x) for x in session_start.split(":"))
    session_open = ts.dt.normalize() + pd.Timedelta(hours=hour, minutes=minute)
    return (ts - session_open).dt.total_seconds() / 60.0


def add_po3_features(bars: pd.DataFrame) -> pd.DataFrame:
    df = bars.copy()
    session_date = pd.to_datetime(df["timestamp"]).dt.date
    minutes_since_open = _minutes_since_open(df["timestamp"])
    in_accum_window = minutes_since_open < ACCUMULATION_WINDOW_MINUTES

    high_masked = df["high"].where(in_accum_window)
    low_masked = df["low"].where(in_accum_window)
    accum_high_expanding = high_masked.groupby(session_date).cummax()
    accum_low_expanding = low_masked.groupby(session_date).cummin()
    accum_high = accum_high_expanding.groupby(session_date).ffill()
    accum_low = accum_low_expanding.groupby(session_date).ffill()

    df["dist_to_accum_high"] = (df["close"] - accum_high) / accum_high
    df["dist_to_accum_low"] = (df["close"] - accum_low) / accum_low

    swept_high_now = (df["high"] > accum_high) & (df["close"] <= accum_high)
    swept_low_now = (df["low"] < accum_low) & (df["close"] >= accum_low)
    # Never flag a sweep while the range itself is still forming (nothing to sweep yet).
    swept_high_now &= ~in_accum_window
    swept_low_now &= ~in_accum_window

    df["swept_accum_high_recent"] = (
        swept_high_now.groupby(session_date)
        .rolling(SWEEP_LOOKBACK_BARS, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
        .astype(float)
    )
    df["swept_accum_low_recent"] = (
        swept_low_now.groupby(session_date)
        .rolling(SWEEP_LOOKBACK_BARS, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
        .astype(float)
    )

    idx = pd.Series(np.arange(len(df)), index=df.index)
    any_sweep = swept_high_now | swept_low_now
    last_sweep_idx = idx.where(any_sweep).groupby(session_date).ffill()
    bars_since = idx - last_sweep_idx
    df["bars_since_sweep"] = bars_since.fillna(RECENCY_CAP_BARS).clip(upper=RECENCY_CAP_BARS)

    sweep_sign = np.where(swept_low_now, 1.0, np.where(swept_high_now, -1.0, np.nan))
    last_sweep_was_low = pd.Series(sweep_sign, index=df.index)
    last_sweep_direction = last_sweep_was_low.groupby(session_date).ffill()
    df["sweep_direction"] = last_sweep_direction.fillna(0.0)

    return df


def add_fvg_features(bars: pd.DataFrame) -> pd.DataFrame:
    df = bars.copy()
    session_date = pd.to_datetime(df["timestamp"]).dt.date
    n = len(df)

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    day = session_date.to_numpy()

    bull_active = np.zeros(n, dtype=bool)
    bull_top = np.full(n, np.nan)
    bull_bottom = np.full(n, np.nan)
    bear_active = np.zeros(n, dtype=bool)
    bear_top = np.full(n, np.nan)
    bear_bottom = np.full(n, np.nan)
    bars_since_bull_ifvg = np.full(n, float(RECENCY_CAP_BARS))
    bars_since_bear_ifvg = np.full(n, float(RECENCY_CAP_BARS))

    cur_bull_top = cur_bull_bottom = np.nan
    cur_bear_top = cur_bear_bottom = np.nan
    cur_bull_active = cur_bear_active = False
    last_bull_ifvg_idx = -np.inf
    last_bear_ifvg_idx = -np.inf
    day_start_idx = 0

    for i in range(n):
        if i > 0 and day[i] != day[i - 1]:
            # New session: reset all gap state, matching PO3's per-session reset.
            cur_bull_active = cur_bear_active = False
            cur_bull_top = cur_bull_bottom = np.nan
            cur_bear_top = cur_bear_bottom = np.nan
            last_bull_ifvg_idx = -np.inf
            last_bear_ifvg_idx = -np.inf
            day_start_idx = i

        # 1. Update existing gaps against this bar's range (fill / invert).
        if cur_bull_active:
            if close[i] < cur_bull_bottom:
                cur_bull_active = False
                last_bull_ifvg_idx = i
            elif low[i] <= cur_bull_bottom:
                cur_bull_active = False
        if cur_bear_active:
            if close[i] > cur_bear_top:
                cur_bear_active = False
                last_bear_ifvg_idx = i
            elif high[i] >= cur_bear_top:
                cur_bear_active = False

        # 2. Detect a new gap forming at this bar (needs i-2, i-1, i within the same session).
        if i - day_start_idx >= 2:
            if high[i - 2] < low[i]:
                cur_bull_active = True
                cur_bull_top, cur_bull_bottom = low[i], high[i - 2]
            if low[i - 2] > high[i]:
                cur_bear_active = True
                cur_bear_top, cur_bear_bottom = low[i - 2], high[i]

        bull_active[i] = cur_bull_active
        bull_top[i], bull_bottom[i] = cur_bull_top, cur_bull_bottom
        bear_active[i] = cur_bear_active
        bear_top[i], bear_bottom[i] = cur_bear_top, cur_bear_bottom
        bars_since_bull_ifvg[i] = min(i - last_bull_ifvg_idx, RECENCY_CAP_BARS)
        bars_since_bear_ifvg[i] = min(i - last_bear_ifvg_idx, RECENCY_CAP_BARS)

    # "No active gap" is itself meaningful (not missing data) — fill with 0.0
    # rather than NaN so it isn't dropped by dataset assembly's dropna, and
    # pair it with the *_active flag so the model can distinguish "0 distance,
    # no gap" from "gap right at the current price" (active=1, dist~0).
    df["bull_fvg_active"] = bull_active.astype(float)
    df["bull_fvg_dist"] = np.where(bull_active, (close - bull_top) / close, 0.0)
    df["bear_fvg_active"] = bear_active.astype(float)
    df["bear_fvg_dist"] = np.where(bear_active, (bear_bottom - close) / close, 0.0)
    df["bars_since_bull_ifvg"] = bars_since_bull_ifvg
    df["bars_since_bear_ifvg"] = bars_since_bear_ifvg

    return df


def add_ict_features(bars: pd.DataFrame) -> pd.DataFrame:
    df = add_po3_features(bars)
    df = add_fvg_features(df)
    return df
