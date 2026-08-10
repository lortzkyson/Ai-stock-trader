from __future__ import annotations

import numpy as np
import pandas as pd


def make_session_minutes(
    day: str, tz: str = "America/New_York", start: str = "09:30", end: str = "16:00"
) -> pd.DatetimeIndex:
    open_ts = pd.Timestamp(f"{day} {start}", tz=tz)
    close_ts = pd.Timestamp(f"{day} {end}", tz=tz)
    return pd.date_range(open_ts, close_ts, freq="1min", inclusive="left")


def make_random_walk_bars(days: list[str], seed: int = 0, bar_std: float = 0.3) -> pd.DataFrame:
    """Bars with enough randomness to actually cross small triple-barrier levels.

    Unlike a smooth deterministic drift, this gives a mix of profit-target,
    stop-loss, and timeout labels, which real training data needs to be a
    meaningful test of the training/validation pipeline.
    """
    rng = np.random.default_rng(seed)
    rows = []
    price = 100.0
    for day in days:
        for ts in make_session_minutes(day):
            price = max(price + rng.normal(0, bar_std), 1.0)
            high = price + abs(rng.normal(0, bar_std * 0.5))
            low = max(price - abs(rng.normal(0, bar_std * 0.5)), 0.5)
            rows.append(
                {
                    "timestamp": ts,
                    "open": price,
                    "high": high,
                    "low": low,
                    "close": price,
                    "volume": int(abs(rng.normal(1_000, 200))) + 1,
                }
            )
    return pd.DataFrame(rows)
