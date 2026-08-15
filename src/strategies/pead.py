"""Post-earnings announcement drift (Ball & Brown 1968; Bernard & Thomas 1989).

The thesis: after a company reports earnings that beat or miss expectations, the
price continues drifting in the direction of the surprise for weeks. It persists
because investors under-react to earnings news, and because limits to arbitrage
(short costs, capital constraints, career risk) stop it being fully traded away.
Like the volatility risk premium and unlike momentum, there is a *mechanism*,
not just a pattern.

Measurement decisions that determine whether the result is honest:

- **The announcement reaction is excluded.** Earnings are typically released
  after the close, so the market's immediate repricing happens on the next
  session. Capturing that would measure a jump nobody could have traded, not
  drift. Entry is at the close of the first full session *after* the reaction,
  so only genuinely subsequent movement is counted.
- **Returns are abnormal, not raw.** A drift measured in a rising market mostly
  measures the rising market. Every event return is net of the benchmark over
  the identical window.
- **Surprise is the sort variable**, taken from reported-vs-estimate rather than
  inferred from price action. Inferring surprise from the price reaction and
  then measuring the subsequent price reaction risks circularity.

Known limitation: the earnings source (Yahoo, via yfinance) only covers live
companies, so delisted names are absent and this analysis is survivorship-biased
even when run against the corrected price universe. The bias is milder here than
for momentum — a 60-day event window doesn't compound over a decade — but it is
not zero, and it cuts the same way: failures are missing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REACTION_DAYS = 1  # sessions consumed by the announcement reaction itself
DEFAULT_DRIFT_DAYS = 60


@dataclass(frozen=True)
class PeadConfig:
    drift_days: int = DEFAULT_DRIFT_DAYS
    reaction_days: int = REACTION_DAYS
    n_buckets: int = 5
    min_events_per_bucket: int = 30


def align_to_trading_day(
    announcement: pd.Timestamp, trading_days: pd.DatetimeIndex
) -> pd.Timestamp | None:
    """First trading session strictly after the announcement timestamp.

    Earnings released after the close on day D are first tradeable on D+1, and
    that session carries the reaction rather than the drift.
    """
    later = trading_days[trading_days > announcement]
    return later[0] if len(later) else None


def compute_event_drift(
    prices: pd.Series,
    benchmark: pd.Series,
    reaction_day: pd.Timestamp,
    config: PeadConfig,
) -> dict | None:
    """Abnormal return over the drift window, excluding the reaction session."""
    idx = prices.index
    if reaction_day not in idx:
        return None
    located = idx.get_loc(reaction_day)
    if not isinstance(located, (int, np.integer)):
        return None  # duplicate timestamps -> ambiguous position, skip the event
    i = int(located)

    entry_i = i + config.reaction_days
    exit_i = entry_i + config.drift_days
    if exit_i >= len(idx):
        return None

    entry_px, exit_px = prices.iloc[entry_i], prices.iloc[exit_i]
    if not np.isfinite(entry_px) or not np.isfinite(exit_px) or entry_px <= 0:
        return None

    stock_return = exit_px / entry_px - 1

    bench = benchmark.reindex(idx)
    b_entry, b_exit = bench.iloc[entry_i], bench.iloc[exit_i]
    if not np.isfinite(b_entry) or not np.isfinite(b_exit) or b_entry <= 0:
        return None
    bench_return = b_exit / b_entry - 1

    # The reaction itself, reported for context but never traded on.
    reaction = prices.iloc[i] / prices.iloc[i - 1] - 1 if i > 0 else np.nan

    return {
        "entry_date": idx[entry_i],
        "exit_date": idx[exit_i],
        "stock_return": float(stock_return),
        "benchmark_return": float(bench_return),
        "abnormal_return": float(stock_return - bench_return),
        "reaction_return": float(reaction) if np.isfinite(reaction) else np.nan,
    }


def compute_sue(events: pd.DataFrame, min_history: int = 6) -> pd.DataFrame:
    """Standardized Unexpected Earnings: surprise divided by the company's own
    historical surprise volatility.

    Percentage surprise is unusable as a sort variable. A company expected to
    earn $0.01 that reports $0.02 shows +100%, dwarfing a genuine blowout at a
    company earning $3.00 — so percentage buckets sort on *small denominators*
    rather than on surprise magnitude. Measured on a first pass, the extreme
    bucket averaged +94.8% surprise and produced non-monotonic drift, which is
    the signature of exactly that.

    SUE is the standard fix (Bernard & Thomas 1989): express each surprise in
    units of how surprising that company's results normally are. Computed from
    strictly prior announcements only — using the full history would leak.
    """
    out = events.sort_values(["symbol", "announced"]).copy()
    out["surprise_abs"] = out["reported_eps"] - out["estimate_eps"]

    grouped = out.groupby("symbol")["surprise_abs"]
    trailing_std = grouped.transform(
        lambda s: s.shift(1).expanding(min_periods=min_history).std()
    )
    trailing_mean = grouped.transform(
        lambda s: s.shift(1).expanding(min_periods=min_history).mean()
    )
    out["sue"] = (out["surprise_abs"] - trailing_mean) / trailing_std
    return out


def bucket_by_surprise(
    events: pd.DataFrame, config: PeadConfig, sort_column: str = "sue"
) -> pd.DataFrame:
    """Rank events into surprise buckets (1 = most negative, n = most positive)."""
    out = events.dropna(subset=[sort_column, "abnormal_return"]).copy()
    out = out[np.isfinite(out[sort_column])]
    if out.empty:
        return out
    out["bucket"] = pd.qcut(
        out[sort_column].rank(method="first"), config.n_buckets,
        labels=range(1, config.n_buckets + 1),
    )
    return out


def summarize_by_bucket(events: pd.DataFrame) -> pd.DataFrame:
    """Mean abnormal drift per surprise bucket, with a t-test against zero."""
    from scipy import stats

    rows = []
    for bucket, group in events.groupby("bucket", observed=True):
        ar = group["abnormal_return"]
        t_stat, p_value = stats.ttest_1samp(ar, 0) if len(ar) > 2 else (np.nan, np.nan)
        rows.append({
            "bucket": bucket,
            "n": len(group),
            "mean_sue": group["sue"].mean(),
            "mean_abnormal_return": ar.mean(),
            "median_abnormal_return": ar.median(),
            "win_rate": (ar > 0).mean(),
            "t_stat": t_stat,
            "p_value": p_value,
        })
    return pd.DataFrame(rows)
