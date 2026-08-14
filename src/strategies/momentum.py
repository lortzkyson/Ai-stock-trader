"""Cross-sectional momentum (Jegadeesh & Titman 1993).

The thesis, and why it's categorically different from what Phases 3-5 tried:
this is a *rules-based* strategy with no fitted parameters, not a model
searched against this dataset. The formation window (12 months), skip window
(1 month), and monthly rebalance are the canonical published values. That
matters for credibility — a result here can't be explained by having tuned
knobs until the backtest looked good, because no knobs were tuned.

**Do not tune these parameters against backtest results.** Trying
formation=6/9/11 months until one looks better is exactly the
iteration-overfitting failure docs/pre-mortem.md §5 exists to prevent, and it
would forfeit the main advantage this approach has over the ML attempts. If a
parameter genuinely needs changing, log the run in experiment_log.csv and say
why in advance.

The skip month is not incidental: short-horizon (1-month) returns exhibit
*reversal*, not continuation, so including the most recent month actively
works against the signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pandas as pd

TRADING_DAYS_PER_MONTH = 21
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class MomentumConfig:
    formation_days: int = TRADING_DAYS_PER_YEAR  # 12 months
    skip_days: int = TRADING_DAYS_PER_MONTH  # 1 month
    n_positions: int = 20
    min_price: float = 5.0
    min_dollar_volume: float = 5_000_000.0
    dollar_volume_window: int = TRADING_DAYS_PER_MONTH


def build_price_matrix(panel: pd.DataFrame, field: str = "close") -> pd.DataFrame:
    """Wide matrix: rows = date, columns = symbol. Much faster for cross-sectional work."""
    matrix = panel.pivot_table(index="timestamp", columns="symbol", values=field, aggfunc="last")
    matrix.index = pd.to_datetime(matrix.index)
    return matrix.sort_index()


def compute_momentum_scores(prices: pd.DataFrame, config: MomentumConfig) -> pd.DataFrame:
    """Momentum score at each date: return from t-formation_days to t-skip_days.

    Uses only shifted (past) prices, so the score at date t is knowable at the
    close of t — no lookahead. Execution still happens after t (see the
    portfolio backtester's next-open fill).
    """
    past = prices.shift(config.skip_days)
    older = prices.shift(config.formation_days)
    return past / older - 1.0


def compute_dollar_volume(panel: pd.DataFrame, config: MomentumConfig) -> pd.DataFrame:
    dollar_volume = panel.assign(dv=panel["close"] * panel["volume"])
    matrix = dollar_volume.pivot_table(
        index="timestamp", columns="symbol", values="dv", aggfunc="last"
    )
    matrix.index = pd.to_datetime(matrix.index)
    return matrix.sort_index().rolling(config.dollar_volume_window).mean()


def month_end_rebalance_dates(prices: pd.DataFrame) -> list[pd.Timestamp]:
    """Last available trading day of each month present in the price index."""
    idx = pd.Series(prices.index, index=prices.index)
    return sorted(idx.groupby([idx.dt.year, idx.dt.month]).max().tolist())


def select_positions(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    config: MomentumConfig,
) -> list[str]:
    """Top-N momentum names that are investable at `rebalance_date`.

    Eligibility (price floor, liquidity floor, sufficient history) is applied
    *before* ranking, so an illiquid microcap with a huge score can't crowd out
    a tradable name.
    """
    if rebalance_date not in scores.index:
        return []

    # Casts: the index is unique per date, so .loc returns a Series, but
    # pandas-stubs can't narrow that from the type alone.
    day_scores = cast(pd.Series, scores.loc[rebalance_date])
    day_prices = cast(pd.Series, prices.loc[rebalance_date])
    day_dv = (
        cast(pd.Series, dollar_volume.loc[rebalance_date])
        if rebalance_date in dollar_volume.index
        else None
    )

    eligible = day_scores.notna() & day_prices.notna() & (day_prices >= config.min_price)
    if day_dv is not None:
        eligible &= day_dv.notna() & (day_dv >= config.min_dollar_volume)

    candidates = day_scores.loc[eligible]
    if candidates.empty:
        return []
    return list(candidates.nlargest(config.n_positions).index)


def build_target_portfolios(
    panel: pd.DataFrame, config: MomentumConfig
) -> tuple[dict[pd.Timestamp, list[str]], pd.DataFrame]:
    """Return {rebalance_date: [symbols to hold]} plus the price matrix used."""
    prices = build_price_matrix(panel, "close")
    scores = compute_momentum_scores(prices, config)
    dollar_volume = compute_dollar_volume(panel, config)

    targets: dict[pd.Timestamp, list[str]] = {}
    for rebalance_date in month_end_rebalance_dates(prices):
        selected = select_positions(scores, prices, dollar_volume, rebalance_date, config)
        if selected:
            targets[rebalance_date] = selected
    return targets, prices


def select_losers(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    config: MomentumConfig,
) -> list[str]:
    """Bottom-N momentum names — the short leg of classic long-short momentum.

    Same eligibility gates as the long leg. In practice the short leg is the
    harder one to actually trade: beaten-down small caps are exactly the names
    that are expensive or impossible to borrow, which the backtester charges for
    but cannot fully capture.
    """
    if rebalance_date not in scores.index:
        return []

    day_scores = cast(pd.Series, scores.loc[rebalance_date])
    day_prices = cast(pd.Series, prices.loc[rebalance_date])
    day_dv = (
        cast(pd.Series, dollar_volume.loc[rebalance_date])
        if rebalance_date in dollar_volume.index
        else None
    )

    eligible = day_scores.notna() & day_prices.notna() & (day_prices >= config.min_price)
    if day_dv is not None:
        eligible &= day_dv.notna() & (day_dv >= config.min_dollar_volume)

    candidates = day_scores.loc[eligible]
    if candidates.empty:
        return []
    return list(candidates.nsmallest(config.n_positions).index)


def build_long_short_portfolios(
    panel: pd.DataFrame, config: MomentumConfig
) -> tuple[dict[pd.Timestamp, list[str]], dict[pd.Timestamp, list[str]], pd.DataFrame]:
    """Return (long targets, short targets, price matrix) for long-short momentum."""
    prices = build_price_matrix(panel, "close")
    scores = compute_momentum_scores(prices, config)
    dollar_volume = compute_dollar_volume(panel, config)

    longs: dict[pd.Timestamp, list[str]] = {}
    shorts: dict[pd.Timestamp, list[str]] = {}
    for rebalance_date in month_end_rebalance_dates(prices):
        winners = select_positions(scores, prices, dollar_volume, rebalance_date, config)
        losers = select_losers(scores, prices, dollar_volume, rebalance_date, config)
        if winners:
            longs[rebalance_date] = winners
        if losers:
            shorts[rebalance_date] = losers
    return longs, shorts, prices
