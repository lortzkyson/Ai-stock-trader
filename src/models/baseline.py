"""Naive baselines the model must beat to be worth using (docs/pre-mortem.md guard #3)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.metrics import compute_signal_metrics


def random_entry_baseline(test_df: pd.DataFrame, n_trades: int, seed: int = 42) -> dict:
    """Same trade count and risk parameters as the model, entries chosen at random."""
    if n_trades == 0 or len(test_df) == 0:
        return compute_signal_metrics(np.array([]), np.array([]), np.array([]))

    rng = np.random.default_rng(seed)
    n_trades = min(n_trades, len(test_df))
    idx = rng.choice(len(test_df), size=n_trades, replace=False)
    sample = test_df.iloc[idx]

    return compute_signal_metrics(
        sample["realized_return"].to_numpy(),
        sample["label"].to_numpy(),
        sample["date"].to_numpy(),
    )


def buy_and_hold_baseline(bars_by_symbol: dict[str, pd.DataFrame]) -> float:
    """Equal-weighted buy-and-hold return across symbols over their combined bar range."""
    returns = []
    for df in bars_by_symbol.values():
        if len(df) < 2:
            continue
        returns.append(df["close"].iloc[-1] / df["close"].iloc[0] - 1)
    return float(np.mean(returns)) if returns else float("nan")
