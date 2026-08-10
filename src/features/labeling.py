"""Triple-barrier labeling (Lopez de Prado). See docs/pre-mortem.md guard #1.

Barrier levels here are the single source of truth for stop-loss/take-profit
sizing: Phase 6's risk engine imports TripleBarrierConfig's defaults so
backtest and live behavior match (docs/pre-mortem.md guard #9).

Within-bar ambiguity (a bar's range crosses both the profit and stop barrier
in the same bar) is resolved conservatively: the stop-loss is assumed to
trigger first. This mirrors the backtester's fill assumption (Phase 5), so
labels and backtest fills use the same conservative rule.

Vectorized with sliding-window views rather than a per-row Python loop:
looping in Python over every bar with a multi-hundred-bar forward horizon is
too slow at real dataset sizes (hundreds of thousands of rows).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


@dataclass(frozen=True)
class TripleBarrierConfig:
    profit_target_pct: float = 0.02
    stop_loss_pct: float = 0.01
    max_holding_bars: int = 390 * 3  # up to 3 regular sessions of 1-minute bars


def label_triple_barrier(
    bars: pd.DataFrame,
    config: TripleBarrierConfig | None = None,
) -> pd.DataFrame:
    """Label every bar as a potential long entry using the triple-barrier method.

    `bars` must be a single symbol's regular-session bars, sorted by
    timestamp ascending (as returned by clean_bars / filter_regular_session).
    Entry price is the bar's own close; barriers are evaluated against the
    following bars' high/low.

    Returns a copy of `bars` with added columns:
      - label: 1 (profit target hit), -1 (stop hit), 0 (timed out), NaN (insufficient horizon)
      - exit_bar_offset: bars from entry to exit
      - realized_return: return booked at exit under the conservative fill assumption
    """
    config = config or TripleBarrierConfig()
    n = len(bars)
    horizon = config.max_holding_bars
    if n == 0:
        return bars.assign(label=pd.Series(dtype=float), exit_bar_offset=pd.Series(dtype=float),
                            realized_return=pd.Series(dtype=float))

    close = bars["close"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)

    # Pad with sentinels that can never satisfy a barrier, so every entry has
    # a full `horizon`-bar window to slide over without bounds-checking.
    high_padded = np.concatenate([high, np.full(horizon, -np.inf)])
    low_padded = np.concatenate([low, np.full(horizon, np.inf)])

    high_windows = sliding_window_view(high_padded[1:], horizon)[:n]
    low_windows = sliding_window_view(low_padded[1:], horizon)[:n]

    profit_target = close * (1 + config.profit_target_pct)
    stop_loss = close * (1 - config.stop_loss_pct)

    profit_hit = high_windows >= profit_target[:, None]
    stop_hit = low_windows <= stop_loss[:, None]

    any_profit = profit_hit.any(axis=1)
    any_stop = stop_hit.any(axis=1)
    first_profit = np.where(any_profit, profit_hit.argmax(axis=1), horizon)
    first_stop = np.where(any_stop, stop_hit.argmax(axis=1), horizon)

    stop_wins = first_stop <= first_profit  # tie -> stop wins (conservative)
    timed_out = (first_profit == horizon) & (first_stop == horizon)

    label = np.where(timed_out, 0, np.where(stop_wins, -1, 1)).astype(float)
    exit_offset = np.where(
        timed_out, horizon, np.where(stop_wins, first_stop + 1, first_profit + 1)
    ).astype(float)

    positions = np.arange(n)
    exit_idx = np.clip(positions + exit_offset.astype(int), 0, n - 1)
    timeout_return = close[exit_idx] / close - 1
    realized_return = np.where(
        timed_out,
        timeout_return,
        np.where(stop_wins, -config.stop_loss_pct, config.profit_target_pct),
    )

    # Rows near the tail don't have a full horizon of *real* forward bars —
    # the sentinel padding would otherwise silently look like a legitimate
    # timeout. Mark those invalid unless a real barrier already triggered
    # within the bars that do exist.
    real_bars_available = n - 1 - positions
    insufficient_horizon = real_bars_available < horizon
    invalid = insufficient_horizon & timed_out

    label[invalid] = np.nan
    exit_offset[invalid] = np.nan
    realized_return[invalid] = np.nan

    labeled = bars.copy()
    labeled["label"] = label
    labeled["exit_bar_offset"] = exit_offset
    labeled["realized_return"] = realized_return
    return labeled


def report_class_balance(labels: pd.Series, skew_threshold: float = 0.10) -> dict:
    """Report label class proportions and flag if any class is under `skew_threshold`."""
    valid = labels.dropna()
    counts = valid.value_counts().sort_index()
    proportions = (counts / len(valid)).to_dict() if len(valid) else {}
    is_skewed = any(p < skew_threshold for p in proportions.values())
    return {
        "counts": counts.to_dict(),
        "proportions": proportions,
        "is_skewed": is_skewed,
        "n_valid": int(len(valid)),
        "n_dropped_insufficient_horizon": int(labels.isna().sum()),
    }
