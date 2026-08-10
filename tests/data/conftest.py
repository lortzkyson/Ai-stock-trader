from __future__ import annotations

import pandas as pd


def make_session_minutes(
    day: str, tz: str = "America/New_York", start: str = "09:30", end: str = "16:00"
) -> pd.DatetimeIndex:
    open_ts = pd.Timestamp(f"{day} {start}", tz=tz)
    close_ts = pd.Timestamp(f"{day} {end}", tz=tz)
    return pd.date_range(open_ts, close_ts, freq="1min", inclusive="left")


def make_clean_bars(days: list[str]) -> pd.DataFrame:
    rows = []
    price = 100.0
    for day in days:
        for ts in make_session_minutes(day):
            price += 0.01
            rows.append(
                {
                    "timestamp": ts,
                    "open": price,
                    "high": price + 0.05,
                    "low": price - 0.05,
                    "close": price,
                    "volume": 1_000,
                }
            )
    return pd.DataFrame(rows)
