"""Data-quality checks: gaps, duplicate timestamps, impossible values, extreme jumps.

Two things confirmed against a real Alpaca pull that synthetic-only testing
couldn't catch: bar timestamps come back UTC (not exchange-local), and the
default response includes pre/post-market bars alongside regular-session
ones. Every function here that reasons about "the trading day" converts to
`session_tz` (default `America/New_York`) first and restricts to the regular
session window before doing so — comparing UTC clock time against a
09:30-16:00 boundary would misclassify nearly every bar.

Halts and extreme-illiquidity days are not available as a labeled feed on
the free plan, so they're detected as a proxy: any regular-session trading
day whose minute bars are missing more than `halt_gap_minutes` of expected
timestamps is flagged as halt-like and excluded by `clean_bars` — a real
trading halt or a too-illiquid-to-fill-reliably day both show up as large
gaps in the minute-bar sequence, and neither produces a fill we can trust in
the backtester (Phase 5).

Extended-hours bars are also dropped by `clean_bars`: they're much thinner
than regular-session volume and would make backtest fills unrealistic (see
docs/pre-mortem.md guard #6), and none of the phases as currently scoped
trade outside 09:30-16:00 ET.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}
DEFAULT_SESSION_TZ = "America/New_York"


@dataclass
class QualityReport:
    symbol: str
    duplicate_timestamp_count: int = 0
    impossible_value_count: int = 0
    extreme_jump_count: int = 0
    halt_like_days: list[pd.Timestamp] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return (
            self.duplicate_timestamp_count == 0
            and self.impossible_value_count == 0
            and not self.halt_like_days
        )


def _require_columns(bars: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(bars.columns)
    if missing:
        raise ValueError(f"bars missing required columns: {sorted(missing)}")


def _to_session_tz(ts: pd.Series, session_tz: str) -> pd.Series:
    if ts.dt.tz is None:
        return ts.dt.tz_localize(session_tz)
    return ts.dt.tz_convert(session_tz)


def find_duplicate_timestamps(bars: pd.DataFrame) -> pd.Series:
    _require_columns(bars)
    return bars["timestamp"].duplicated(keep="first")


def find_impossible_values(bars: pd.DataFrame) -> pd.Series:
    _require_columns(bars)
    non_positive = (bars[["open", "high", "low", "close"]] <= 0).any(axis=1)
    inverted_range = bars["high"] < bars["low"]
    negative_volume = bars["volume"] < 0
    return non_positive | inverted_range | negative_volume


def find_extreme_jumps(bars: pd.DataFrame, max_single_bar_return: float = 0.5) -> pd.Series:
    _require_columns(bars)
    returns = bars["close"].pct_change().abs()
    return returns > max_single_bar_return


def filter_regular_session(
    bars: pd.DataFrame,
    session_start: str = "09:30",
    session_end: str = "16:00",
    session_tz: str = DEFAULT_SESSION_TZ,
) -> pd.DataFrame:
    """Keep only bars within the regular trading session, dropping pre/post-market."""
    _require_columns(bars)
    if bars.empty:
        return bars.copy()

    local_ts = _to_session_tz(pd.to_datetime(bars["timestamp"]), session_tz)
    start_t = pd.Timestamp(session_start).time()
    end_t = pd.Timestamp(session_end).time()
    mask = (local_ts.dt.time >= start_t) & (local_ts.dt.time < end_t)
    return bars.loc[mask].reset_index(drop=True)


def find_halt_like_days(
    bars: pd.DataFrame,
    session_start: str = "09:30",
    session_end: str = "16:00",
    session_tz: str = DEFAULT_SESSION_TZ,
    halt_gap_minutes: int = 5,
) -> list[pd.Timestamp]:
    """Flag regular-session trading days missing more than `halt_gap_minutes` of bars."""
    _require_columns(bars)
    if bars.empty:
        return []

    session_bars = filter_regular_session(bars, session_start, session_end, session_tz)
    if session_bars.empty:
        return []
    local_ts = _to_session_tz(pd.to_datetime(session_bars["timestamp"]), session_tz)

    flagged: list[pd.Timestamp] = []
    for day, day_ts in local_ts.groupby(local_ts.dt.date):
        session_open = pd.Timestamp(f"{day} {session_start}", tz=session_tz)
        session_close = pd.Timestamp(f"{day} {session_end}", tz=session_tz)
        expected = pd.date_range(session_open, session_close, freq="1min", inclusive="left")
        missing = expected.difference(pd.DatetimeIndex(day_ts))
        if len(missing) > halt_gap_minutes:
            flagged.append(pd.Timestamp(day))
    return flagged


def check_quality(
    bars: pd.DataFrame,
    symbol: str,
    max_single_bar_return: float = 0.5,
    session_start: str = "09:30",
    session_end: str = "16:00",
    session_tz: str = DEFAULT_SESSION_TZ,
    halt_gap_minutes: int = 5,
) -> QualityReport:
    _require_columns(bars)
    return QualityReport(
        symbol=symbol,
        duplicate_timestamp_count=int(find_duplicate_timestamps(bars).sum()),
        impossible_value_count=int(find_impossible_values(bars).sum()),
        extreme_jump_count=int(find_extreme_jumps(bars, max_single_bar_return).sum()),
        halt_like_days=find_halt_like_days(
            bars, session_start, session_end, session_tz, halt_gap_minutes
        ),
    )


def clean_bars(
    bars: pd.DataFrame,
    report: QualityReport,
    session_start: str = "09:30",
    session_end: str = "16:00",
    session_tz: str = DEFAULT_SESSION_TZ,
) -> pd.DataFrame:
    """Drop duplicate timestamps, impossible-value rows, extended-hours bars, and halt-like days.

    Extreme single-bar jumps are flagged in the report but not dropped here —
    a large move can be a legitimate news/earnings gap, not a data error, so
    it's left for manual review rather than silently removed.
    """
    _require_columns(bars)
    keep = ~find_duplicate_timestamps(bars) & ~find_impossible_values(bars)
    cleaned = bars.loc[keep].reset_index(drop=True)

    cleaned = filter_regular_session(cleaned, session_start, session_end, session_tz)

    if report.halt_like_days:
        halt_dates = {d.date() for d in report.halt_like_days}
        local_ts = _to_session_tz(pd.to_datetime(cleaned["timestamp"]), session_tz)
        cleaned = cleaned.loc[~local_ts.dt.date.isin(halt_dates)].reset_index(drop=True)

    return cleaned
