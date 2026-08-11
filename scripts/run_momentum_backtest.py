#!/usr/bin/env python3
"""Cross-sectional momentum backtest: screen -> fetch -> rank -> rebalance -> report.

Deliberately parameter-free where it counts: the 12-1 formation/skip windows
and monthly rebalance are canonical published values, not tuned here. See
src/strategies/momentum.py for why that matters and why they shouldn't be
tuned against these results.

The holdout window (docs/pre-mortem.md §5) is excluded by default. Run with
--include-holdout only when deliberately spending that one-time test.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from backtest.portfolio import PortfolioBacktestConfig, run_portfolio_backtest  # noqa: E402
from backtest.portfolio_metrics import (  # noqa: E402
    benchmark_series,
    compare_to_benchmark,
    compute_portfolio_metrics,
    equal_weight_buy_and_hold,
    volatility_matched_benchmark,
)
from data.daily_bars import (  # noqa: E402
    check_daily_quality,
    clean_daily_panel,
    load_or_fetch_daily_panel,
)
from data.holdout import HOLDOUT_END, HOLDOUT_START  # noqa: E402
from data.universe_screen import (  # noqa: E402
    ScreenConfig,
    build_union_universe,
    list_candidate_symbols,
)
from strategies.momentum import MomentumConfig, build_target_portfolios  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
START = date(2016, 1, 1)
END = date(2026, 4, 30)  # holdout begins 2026-05-01
SCREEN_DATES = [date(2017, 6, 30), date(2020, 6, 30), date(2023, 6, 30), date(2025, 12, 31)]
BENCHMARK_SYMBOL = "SPY"
STARTING_EQUITY = 10_000.0

UNIVERSE_CACHE = REPO_ROOT / "data" / "cache" / "momentum_universe.txt"


def fmt(x: float) -> str:
    return "nan" if x != x else f"{x:.4f}"


def load_universe(client: StockHistoricalDataClient, refresh: bool) -> list[str]:
    if UNIVERSE_CACHE.exists() and not refresh:
        return [s for s in UNIVERSE_CACHE.read_text().split() if s]

    print("Listing candidate symbols (NYSE + NASDAQ, tradable common stock)...")
    candidates = list_candidate_symbols()
    print(f"  {len(candidates)} candidates")

    print(f"Screening liquidity at {len(SCREEN_DATES)} points in time (union)...")
    universe = build_union_universe(
        client, candidates, SCREEN_DATES, ScreenConfig(), progress=True
    )
    UNIVERSE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    UNIVERSE_CACHE.write_text("\n".join(universe))
    return universe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-universe", action="store_true")
    parser.add_argument(
        "--include-holdout",
        action="store_true",
        help="Spend the one-time holdout test (docs/pre-mortem.md §5). Think first.",
    )
    args = parser.parse_args()

    load_dotenv()
    client = StockHistoricalDataClient(
        os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]
    )

    universe = load_universe(client, args.refresh_universe)
    print(f"Universe: {len(universe)} symbols")

    end = HOLDOUT_END if args.include_holdout else END
    fetch_symbols = sorted(set(universe) | {BENCHMARK_SYMBOL})
    print(f"Fetching daily bars {START}..{end} for {len(fetch_symbols)} symbols...")
    panel = load_or_fetch_daily_panel(client, fetch_symbols, START, end, progress=True)
    print(f"  {len(panel):,} rows")

    quality = check_daily_quality(panel)
    print(f"Quality: {quality}")
    panel = clean_daily_panel(panel)

    if not args.include_holdout:
        ts = pd.to_datetime(panel["timestamp"]).dt.date
        in_holdout = (ts >= HOLDOUT_START) & (ts <= HOLDOUT_END)
        panel = panel.loc[~in_holdout].reset_index(drop=True)
        print(f"  {len(panel):,} rows after excluding holdout window")

    momentum_config = MomentumConfig()
    print(
        f"Building momentum targets (formation={momentum_config.formation_days}d, "
        f"skip={momentum_config.skip_days}d, n={momentum_config.n_positions})..."
    )
    strategy_panel = panel.loc[panel["symbol"] != BENCHMARK_SYMBOL]
    targets, close_prices = build_target_portfolios(strategy_panel, momentum_config)
    print(f"  {len(targets)} rebalance dates")

    open_prices = strategy_panel.pivot_table(
        index="timestamp", columns="symbol", values="open", aggfunc="last"
    ).sort_index()
    open_prices.index = pd.to_datetime(open_prices.index)

    print("Running portfolio backtest...")
    result = run_portfolio_backtest(
        targets, close_prices, open_prices,
        PortfolioBacktestConfig(starting_equity=STARTING_EQUITY),
    )
    strategy_metrics = compute_portfolio_metrics(result.daily_equity)
    print(f"Strategy: {strategy_metrics}")

    all_close = (
        panel.pivot_table(index="timestamp", columns="symbol", values="close", aggfunc="last")
        .sort_index()
    )
    all_close.index = pd.to_datetime(all_close.index)

    spy_equity = benchmark_series(all_close, BENCHMARK_SYMBOL, STARTING_EQUITY)
    universe_equity = equal_weight_buy_and_hold(all_close, universe, STARTING_EQUITY)

    levered_spy, leverage = volatility_matched_benchmark(
        result.daily_equity, spy_equity, STARTING_EQUITY
    )

    spy_metrics = compute_portfolio_metrics(spy_equity)
    universe_metrics = compute_portfolio_metrics(universe_equity)
    levered_metrics = compute_portfolio_metrics(levered_spy)
    vs_spy = compare_to_benchmark(result.daily_equity, spy_equity)
    vs_universe = compare_to_benchmark(result.daily_equity, universe_equity)
    vs_levered = compare_to_benchmark(result.daily_equity, levered_spy)

    print(f"SPY buy-and-hold: {spy_metrics}")
    print(f"Equal-weight universe buy-and-hold: {universe_metrics}")
    print(f"Vol-matched SPY ({leverage:.2f}x): {levered_metrics}")
    print(f"vs SPY: {vs_spy}")
    print(f"vs universe: {vs_universe}")
    print(f"vs VOL-MATCHED SPY (the benchmark that matters): {vs_levered}")

    write_report(
        universe, panel, targets, result, strategy_metrics,
        spy_metrics, universe_metrics, levered_metrics,
        vs_spy, vs_universe, vs_levered, leverage,
        momentum_config, end, args.include_holdout,
    )
    return 0


def write_report(
    universe, panel, targets, result, strategy_metrics,
    spy_metrics, universe_metrics, levered_metrics,
    vs_spy, vs_universe, vs_levered, leverage,
    momentum_config, end, include_holdout,
) -> None:
    stamp = datetime.now(timezone.utc)
    suffix = "_HOLDOUT" if include_holdout else ""
    path = REPO_ROOT / "reports" / f"momentum_{stamp:%Y-%m-%d_%H%M%S}{suffix}.md"

    lines = [f"# Cross-Sectional Momentum Backtest ({stamp:%Y-%m-%d})\n"]
    lines.append(
        "Canonical Jegadeesh-Titman 12-1 momentum: rank by return from t-12mo to t-1mo, "
        "hold the top N equal-weighted, rebalance monthly. **No parameters were tuned "
        "against this data** — the formation/skip windows and rebalance frequency are the "
        "published values. That's the main methodological advantage over the Phase 3-5 "
        "intraday attempts, and it's forfeited the moment anyone starts tweaking them to "
        "improve these numbers.\n"
    )

    lines.append("## Setup\n")
    holdout_note = " **(HOLDOUT INCLUDED)**" if include_holdout else " (holdout excluded)"
    lines.append(
        f"- Universe: {len(universe)} symbols (liquidity-screened, union of "
        f"{len(SCREEN_DATES)} point-in-time screens)"
    )
    lines.append(f"- Period: {START} to {end}{holdout_note}")
    lines.append(f"- Rebalances: {len(targets)}")
    lines.append(f"- Positions held: {momentum_config.n_positions}, equal-weighted")
    lines.append(f"- Starting equity: ${STARTING_EQUITY:,.0f}")
    lines.append(
        "- Execution: signal at month-end close, fill at next session's open, "
        "5bps adverse slippage, real Alpaca sell fees\n"
    )

    lines.append("## Results vs. benchmarks\n")
    lines.append("| | total return | CAGR | ann. vol | Sharpe | Sortino | max drawdown |")
    lines.append("|---|---|---|---|---|---|---|")
    for label, m in [
        ("**Momentum strategy**", strategy_metrics),
        ("SPY buy-and-hold", spy_metrics),
        ("Equal-weight universe buy-and-hold", universe_metrics),
        (f"SPY levered to match strategy vol ({leverage:.2f}x)", levered_metrics),
    ]:
        lines.append(
            f"| {label} | {fmt(m['total_return'])} | {fmt(m['cagr'])} "
            f"| {fmt(m['annual_volatility'])} | {fmt(m['sharpe'])} "
            f"| {fmt(m['sortino'])} | {fmt(m['max_drawdown'])} |"
        )
    lines.append("")

    lines.append("## Does it beat the benchmark by more than noise?\n")
    lines.append(
        "Paired t-test on daily return differences. In a rising market almost any long "
        "strategy makes money, so 'did it profit' is the wrong question — 'did it beat "
        "buying and holding, beyond chance' is the right one.\n"
    )
    lines.append("| benchmark | excess total return | t-stat | p-value | significant? |")
    lines.append("|---|---|---|---|---|")
    for label, c in [
        ("SPY", vs_spy),
        ("Equal-weight universe", vs_universe),
        (f"**SPY vol-matched ({leverage:.2f}x)**", vs_levered),
    ]:
        verdict = "**yes**" if c["is_significant"] else "no"
        lines.append(
            f"| {label} | {fmt(c['excess_total_return'])} | {fmt(c['t_stat'])} "
            f"| {fmt(c['p_value'])} | {verdict} |"
        )
    lines.append("")
    lines.append(
        f"**The vol-matched row is the one that counts.** This strategy runs at "
        f"{fmt(strategy_metrics['annual_volatility'])} annualized volatility versus SPY's "
        f"{fmt(spy_metrics['annual_volatility'])} — it takes {leverage:.2f}x the risk. A "
        "portfolio taking that much more risk should earn proportionally more *with no skill "
        "whatsoever*, so beating unlevered SPY is not evidence of anything. Only the excess "
        "over a risk-matched benchmark is a candidate for edge.\n"
    )

    if not result.turnover.empty:
        lines.append(f"Average monthly turnover: {result.turnover.mean():.1%}\n")
    if not result.trades.empty:
        lines.append(f"Total rebalance transactions: {len(result.trades)}")
        lines.append(f"Total fees paid: ${result.trades['fees'].sum():,.2f}\n")

    lines.append("## Known limitations\n")
    lines.append(
        "- **Survivorship bias remains.** The universe is built from *currently* tradable "
        "symbols; companies that delisted, went bankrupt, or were acquired during the period "
        "are absent. Alpaca's asset list doesn't include them (its 'inactive' list is OTC "
        "names, not delisted large caps), though their price history *is* queryable if a "
        "point-in-time constituent list is ever sourced. For long-only top-decile momentum "
        "the direction of this bias is ambiguous — excluded failures would inflate results, "
        "excluded acquisitions (often at a premium) would deflate them — but it is not zero."
    )
    lines.append(
        "- Point-in-time liquidity is enforced at each rebalance (trailing 21-day dollar "
        "volume), but the *candidate* list was screened at four fixed dates, which is an "
        "approximation of a true point-in-time universe."
    )
    lines.append(
        "- Delistings mid-hold are handled by dropping the position at the last available "
        "price rather than modeling an acquisition payout or bankruptcy recovery."
    )
    lines.append(
        "- No short leg. Classic momentum research is long-short; this is long-only, which "
        "captures less of the documented effect and carries full market beta."
    )

    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote report to {path}")


if __name__ == "__main__":
    raise SystemExit(main())
