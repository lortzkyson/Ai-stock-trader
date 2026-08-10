"""Data-quality checks: gaps, duplicate timestamps, impossible values, extreme jumps.

Halts and extreme-illiquidity days are not available as a labeled feed on
the free plan, so they're detected as a proxy: any regular-session trading
day whose minute bars are missing more than `halt_gap_minutes` of expected
timestamps is flagged as halt-like and excluded by `clean_bars` — a real
trading halt or a too-illiquid-to-fill-reliably day both show up as large
gaps in the minute-bar sequence, and neither produces a fill we can trust in
the backtester (Phase 5).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}


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


def find_halt_like_days(
    bars: pd.DataFrame,
    session_start: str = "09:30",
    session_end: str = "16:00",
    halt_gap_minutes: int = 5,
) -> list[pd.Timestamp]:
    """Flag trading days missing more than `halt_gap_minutes` of expected minute bars."""
    _require_columns(bars)
    if bars.empty:
        return []

    ts = pd.to_datetime(bars["timestamp"])
    flagged: list[pd.Timestamp] = []
    for day, day_ts in ts.groupby(ts.dt.date):
        session_open = pd.Timestamp.combine(day, pd.Timestamp(session_start).time()).tz_localize(
            day_ts.dt.tz
        )
        session_close = pd.Timestamp.combine(day, pd.Timestamp(session_end).time()).tz_localize(
            day_ts.dt.tz
        )
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
    halt_gap_minutes: int = 5,
) -> QualityReport:
    _require_columns(bars)
    return QualityReport(
        symbol=symbol,
        duplicate_timestamp_count=int(find_duplicate_timestamps(bars).sum()),
        impossible_value_count=int(find_impossible_values(bars).sum()),
        extreme_jump_count=int(find_extreme_jumps(bars, max_single_bar_return).sum()),
        halt_like_days=find_halt_like_days(bars, session_start, session_end, halt_gap_minutes),
    )


def clean_bars(
    bars: pd.DataFrame,
    report: QualityReport,
) -> pd.DataFrame:
    """Drop duplicate timestamps, impossible-value rows, and halt-like days.

    Extreme single-bar jumps are flagged in the report but not dropped here —
    a large move can be a legitimate news/earnings gap, not a data error, so
    it's left for manual review rather than silently removed.
    """
    _require_columns(bars)
    ts = pd.to_datetime(bars["timestamp"])
    keep = ~find_duplicate_timestamps(bars) & ~find_impossible_values(bars)
    if report.halt_like_days:
        halt_dates = {d.date() for d in report.halt_like_days}
        keep &= ~ts.dt.date.isin(halt_dates)
    return bars.loc[keep].reset_index(drop=True)
