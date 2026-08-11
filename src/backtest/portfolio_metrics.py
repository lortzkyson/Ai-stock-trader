"""Portfolio-level performance metrics and benchmarks.

Separate from `backtest/metrics.py` because that module is built around
round-trip trades (win rate, expectancy per trade), which don't map cleanly
onto a rebalancing portfolio where a "trade" is a weight adjustment rather
than a discrete bet. What matters here is the equity curve and, critically,
how it compares to simply having bought and held.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS_PER_YEAR = 252


def compute_portfolio_metrics(daily_equity: pd.Series) -> dict:
    if len(daily_equity) < 2:
        return {k: float("nan") for k in
                ["total_return", "cagr", "annual_volatility", "sharpe", "sortino",
                 "max_drawdown", "n_days"]}

    returns = daily_equity.pct_change().dropna()
    total_return = float(daily_equity.iloc[-1] / daily_equity.iloc[0] - 1)
    years = len(daily_equity) / TRADING_DAYS_PER_YEAR
    growth = daily_equity.iloc[-1] / daily_equity.iloc[0]
    cagr = float(growth ** (1 / years) - 1) if years > 0 else float("nan")

    annual_vol = float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe = (
        float(returns.mean() / returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        if returns.std() > 0
        else float("nan")
    )

    downside = returns[returns < 0]
    sortino = (
        float(returns.mean() / downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        if len(downside) > 1 and downside.std() > 0
        else float("nan")
    )

    running_max = daily_equity.cummax()
    drawdown = daily_equity / running_max - 1
    max_drawdown = float(drawdown.min())

    return {
        "total_return": total_return,
        "cagr": cagr,
        "annual_volatility": annual_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "n_days": int(len(daily_equity)),
    }


def equal_weight_buy_and_hold(
    close_prices: pd.DataFrame, symbols: list[str], starting_equity: float = 10_000.0
) -> pd.Series:
    """Equal-weight buy-and-hold of `symbols`, bought on the first day they all price.

    Names that later delist simply stop contributing (their price goes NaN and
    the last known value is carried); this is a simplification but errs toward
    *flattering* the benchmark, which is the safe direction for a comparison
    the strategy needs to beat.
    """
    available = [s for s in symbols if s in close_prices.columns]
    if not available:
        return pd.Series(dtype=float)

    prices = close_prices[available]
    first_valid = prices.dropna(how="all").index[0]
    entry_prices = prices.loc[first_valid]
    investable = entry_prices.dropna()
    if investable.empty:
        return pd.Series(dtype=float)

    value_each = starting_equity / len(investable)
    shares = value_each / investable

    holdings_value = (prices[investable.index] * shares).ffill()
    return holdings_value.sum(axis=1)


def benchmark_series(
    close_prices: pd.DataFrame, symbol: str, starting_equity: float = 10_000.0
) -> pd.Series:
    """Buy-and-hold a single symbol (e.g. SPY) normalized to `starting_equity`."""
    if symbol not in close_prices.columns:
        return pd.Series(dtype=float)
    series = close_prices[symbol].ffill().dropna()
    if series.empty:
        return pd.Series(dtype=float)
    return starting_equity * series / series.iloc[0]


def volatility_matched_benchmark(
    strategy_equity: pd.Series, benchmark_equity: pd.Series, starting_equity: float = 10_000.0
) -> tuple[pd.Series, float]:
    """Benchmark scaled to the strategy's realized volatility.

    This is the benchmark that actually matters, and omitting it is how
    high-volatility strategies get mistaken for skillful ones. A portfolio with
    2x the market's volatility should earn roughly 2x the market's return
    *without any skill at all* — so beating an unlevered benchmark proves
    nothing on its own. Comparing against a benchmark levered to the same
    volatility isolates whatever return is left over after accounting for
    risk-taking.

    Returns (levered equity curve, leverage factor). Borrowing costs are not
    modeled, which flatters the *benchmark* — the conservative direction when
    the strategy is the thing under scrutiny.
    """
    aligned = pd.DataFrame({"s": strategy_equity, "b": benchmark_equity}).dropna()
    if len(aligned) < 3:
        return pd.Series(dtype=float), float("nan")

    returns = aligned.pct_change().dropna()
    strat_vol, bench_vol = returns["s"].std(), returns["b"].std()
    if bench_vol == 0:
        return pd.Series(dtype=float), float("nan")

    leverage = float(strat_vol / bench_vol)
    levered_returns = returns["b"] * leverage
    equity = starting_equity * (1 + levered_returns).cumprod()
    return equity, leverage


def compare_to_benchmark(strategy_equity: pd.Series, benchmark_equity: pd.Series) -> dict:
    """Excess return and a paired t-test on daily return differences.

    The t-test asks whether the strategy's daily returns differ from the
    benchmark's by more than noise — a much more honest question than "did the
    strategy make money", since in a rising market almost anything does.
    """
    aligned = pd.DataFrame(
        {"strategy": strategy_equity, "benchmark": benchmark_equity}
    ).dropna()
    if len(aligned) < 3:
        return {"n_days": len(aligned), "excess_total_return": float("nan"),
                "t_stat": float("nan"), "p_value": float("nan"), "is_significant": False}

    strat_returns = aligned["strategy"].pct_change().dropna()
    bench_returns = aligned["benchmark"].pct_change().dropna()
    diff = strat_returns - bench_returns

    t_stat, p_value = stats.ttest_1samp(diff, 0)
    strat_total = aligned["strategy"].iloc[-1] / aligned["strategy"].iloc[0] - 1
    bench_total = aligned["benchmark"].iloc[-1] / aligned["benchmark"].iloc[0] - 1

    return {
        "n_days": int(len(aligned)),
        "strategy_total_return": float(strat_total),
        "benchmark_total_return": float(bench_total),
        "excess_total_return": float(strat_total - bench_total),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "is_significant": bool(p_value < 0.05),
    }
