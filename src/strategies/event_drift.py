"""Post-event drift: underreaction to large information events.

Post-earnings-announcement drift (PEAD) is the best-documented anomaly in the
equity literature — prices continue drifting in the direction of an earnings
surprise for weeks afterward, attributed to investor underreaction plus limits
to arbitrage. Testing it directly needs an earnings calendar with consensus
estimates.

That data is not obtainable here without reintroducing survivorship bias: free
sources (yfinance) return nothing for delisted tickers — SIVB, FRC, ATVI, TWTR
and RAD all come back empty — and PEAD is *especially* sensitive to that gap,
since a company that misses badly and then fails disappears from the sample
entirely. Having already measured survivorship bias inflate a momentum backtest
from 12.9% to 35.2% CAGR, importing it again would be indefensible.

So this module tests the *mechanism* rather than the specific event. A large
abnormal move on abnormal volume is an information event — earnings, guidance,
FDA decision, M&A. Detecting those from price and volume alone means the study
runs entirely on the survivorship-corrected panel, with no external data and no
hidden sample truncation.

What it gives up: the events are unlabelled (an earnings beat and a merger
rumour look alike) and the *surprise* is measured by the market's own reaction
rather than against analyst expectations. If drift exists, this finds it. It
just can't attribute it to earnings specifically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class EventDriftConfig:
    # An event is a move large relative to the stock's own recent volatility,
    # on volume well above its own recent norm. Both are relative to the name
    # itself, so a quiet utility and a volatile biotech are judged on their own
    # scales rather than an absolute threshold favouring high-beta names.
    return_sigma_threshold: float = 3.0
    volume_multiple_threshold: float = 2.0
    volatility_lookback: int = 60
    volume_lookback: int = 60

    holding_days: int = 40  # drift window; PEAD literature reports ~60 calendar days
    n_positions: int = 20
    min_price: float = 5.0
    min_dollar_volume: float = 5_000_000.0


def detect_events(panel: pd.DataFrame, config: EventDriftConfig) -> pd.DataFrame:
    """Flag abnormal-return-on-abnormal-volume days.

    Everything is computed from trailing windows shifted by one day, so an
    event is identifiable at that day's close without using its own bar to set
    the threshold it must clear.
    """
    df = panel.sort_values(["symbol", "timestamp"]).reset_index(drop=True).copy()
    grouped = df.groupby("symbol")

    df["ret"] = grouped["close"].pct_change(fill_method=None)
    trailing_vol = grouped["ret"].transform(
        lambda s: s.shift(1).rolling(config.volatility_lookback).std()
    )
    trailing_volume = grouped["volume"].transform(
        lambda s: s.shift(1).rolling(config.volume_lookback).mean()
    )

    df["ret_sigma"] = df["ret"] / trailing_vol
    df["volume_multiple"] = df["volume"] / trailing_volume
    df["dollar_volume"] = df["close"] * df["volume"]

    df["is_event"] = (
        (df["ret_sigma"].abs() >= config.return_sigma_threshold)
        & (df["volume_multiple"] >= config.volume_multiple_threshold)
        & (df["close"] >= config.min_price)
        & (df["dollar_volume"] >= config.min_dollar_volume)
        & trailing_vol.notna()
        & trailing_volume.notna()
    )
    return df


def build_drift_targets(
    panel: pd.DataFrame, config: EventDriftConfig, direction: str = "positive"
) -> dict[pd.Timestamp, list[str]]:
    """Rebalance targets holding names that recently had a large event.

    `direction="positive"` buys names that jumped (the PEAD long leg — drift is
    expected to continue upward). `"negative"` buys names that crashed, which
    tests reversal rather than drift and should *lose* if underreaction is the
    true mechanism — a useful falsification check.
    """
    events = detect_events(panel, config)
    hits = events.loc[events["is_event"]].copy()
    if direction == "positive":
        hits = hits.loc[hits["ret_sigma"] > 0]
    else:
        hits = hits.loc[hits["ret_sigma"] < 0]
    if hits.empty:
        return {}

    all_dates = sorted(pd.to_datetime(panel["timestamp"]).unique())
    date_index = {d: i for i, d in enumerate(all_dates)}

    # A name is held for `holding_days` sessions after its event fires.
    held_on: dict[pd.Timestamp, list[tuple[float, str]]] = {}
    for row_raw in hits.itertuples(index=False):
        row: Any = row_raw  # pandas-stubs types itertuples rows too loosely
        event_date = pd.Timestamp(row.timestamp)
        i = date_index.get(event_date)
        if i is None:
            continue
        strength = abs(float(row.ret_sigma))
        for j in range(i + 1, min(i + 1 + config.holding_days, len(all_dates))):
            held_on.setdefault(all_dates[j], []).append((strength, str(row.symbol)))

    # Rebalance monthly, matching the momentum strategy so results are comparable.
    targets: dict[pd.Timestamp, list[str]] = {}
    dates_series = pd.Series(all_dates, index=all_dates)
    month_ends = sorted(dates_series.groupby([dates_series.dt.year, dates_series.dt.month]).max())
    for rebalance_date in month_ends:
        candidates = held_on.get(rebalance_date, [])
        if not candidates:
            continue
        # Strongest events first; de-duplicate a name appearing twice.
        seen: set[str] = set()
        picked: list[str] = []
        for _, symbol in sorted(candidates, key=lambda x: -x[0]):
            if symbol in seen:
                continue
            seen.add(symbol)
            picked.append(symbol)
            if len(picked) >= config.n_positions:
                break
        if picked:
            targets[rebalance_date] = picked
    return targets


def summarize_events(panel: pd.DataFrame, config: EventDriftConfig) -> dict:
    events = detect_events(panel, config)
    hits = events.loc[events["is_event"]]
    return {
        "n_events": int(len(hits)),
        "n_symbols_with_events": int(hits["symbol"].nunique()),
        "positive_events": int((hits["ret_sigma"] > 0).sum()),
        "negative_events": int((hits["ret_sigma"] < 0).sum()),
        "median_event_return": float(hits["ret"].median()) if len(hits) else float("nan"),
        "events_per_symbol_per_year": (
            float(len(hits) / max(hits["symbol"].nunique(), 1) / 10.3) if len(hits) else 0.0
        ),
    }


def forward_return_study(
    panel: pd.DataFrame, config: EventDriftConfig, horizons: tuple[int, ...] = (5, 10, 20, 40, 60)
) -> pd.DataFrame:
    """Average forward return after an event, versus after a non-event day.

    This is the cleanest test of whether drift exists at all, independent of any
    portfolio construction, position sizing or trading-cost assumption. If there
    is no gap here, no amount of portfolio engineering will manufacture one.
    """
    events = detect_events(panel, config)
    events = events.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    close = events.groupby("symbol")["close"]

    rows = []
    for h in horizons:
        fwd = close.transform(lambda s, h=h: s.shift(-h) / s - 1)
        pos = events["is_event"] & (events["ret_sigma"] > 0)
        neg = events["is_event"] & (events["ret_sigma"] < 0)
        base = ~events["is_event"] & events["ret_sigma"].notna()
        rows.append(
            {
                "horizon_days": h,
                "after_positive_event": float(fwd[pos].mean()),
                "after_negative_event": float(fwd[neg].mean()),
                "baseline_non_event": float(fwd[base].mean()),
                "n_positive": int(pos.sum()),
                "n_negative": int(neg.sum()),
            }
        )
    out = pd.DataFrame(rows)
    out["positive_edge_vs_baseline"] = out["after_positive_event"] - out["baseline_non_event"]
    out["negative_edge_vs_baseline"] = out["after_negative_event"] - out["baseline_non_event"]
    return out


