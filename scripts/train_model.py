#!/usr/bin/env python3
"""Phase 4: build the dataset, walk-forward validate, compare to baselines,
check regime robustness, train the final production model, and write the
model card + experiment log entry.

Run scripts/fetch_training_data.py first to populate the cache.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.cache import read_cached  # noqa: E402
from data.holdout import exclude_holdout  # noqa: E402
from data.quality import check_quality, clean_bars  # noqa: E402
from features.engineering import FEATURE_COLUMNS  # noqa: E402
from features.labeling import TripleBarrierConfig, report_class_balance  # noqa: E402
from models.baseline import buy_and_hold_baseline, random_entry_baseline  # noqa: E402
from models.dataset import build_dataset  # noqa: E402
from models.experiment_log import append_run  # noqa: E402
from models.train import (  # noqa: E402
    aggregate_oos_metrics,
    regime_split_metrics,
    run_walk_forward,
    train_fold_model,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SYMBOLS = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "JPM", "WMT", "XOM"]
START = date(2025, 9, 1)
END = date(2026, 4, 30)
BARRIER_CONFIG = TripleBarrierConfig()  # defaults: 2% target, 1% stop, 3-session horizon
N_FOLDS = 5
EMBARGO_DAYS = 4
PROBABILITY_THRESHOLD = 0.5

MODEL_PATH = REPO_ROOT / "data" / "models" / "production_model.joblib"
MODEL_CARD_PATH = REPO_ROOT / "docs" / "model_card.md"


def fmt(x: float) -> str:
    return "nan" if x != x else f"{x:.4f}"  # noqa: PLR0124 (NaN check without importing numpy)


def main() -> int:
    print(f"Building dataset: {SYMBOLS} {START}..{END}")
    dataset = build_dataset(SYMBOLS, START, END, barrier_config=BARRIER_CONFIG)
    print(f"Dataset rows: {len(dataset)}")

    balance = report_class_balance(dataset["label"])
    print("Class balance:", balance)

    print(f"Running walk-forward validation: {N_FOLDS} folds, embargo={EMBARGO_DAYS} days")
    fold_metrics, oos = run_walk_forward(
        dataset, n_folds=N_FOLDS, embargo_days=EMBARGO_DAYS, probability_threshold=PROBABILITY_THRESHOLD
    )

    print("\n=== Per-fold metrics ===")
    for fm in fold_metrics:
        print(
            f"fold {fm.fold_id}: train={fm.train_start}..{fm.train_end} "
            f"test={fm.test_start}..{fm.test_end} n_train={fm.n_train} n_test={fm.n_test} "
            f"n_trades={fm.n_trades} accuracy={fmt(fm.accuracy)} win_rate={fmt(fm.win_rate)} "
            f"expectancy={fmt(fm.expectancy)} sharpe={fmt(fm.sharpe)} max_dd={fmt(fm.max_drawdown)}"
        )

    agg = aggregate_oos_metrics(oos)
    print("\n=== Aggregate out-of-sample (all folds) ===")
    print(agg)

    # Baselines, evaluated on the same aggregate out-of-sample trade count for a fair comparison.
    total_trades = agg["n_trades"]
    random_baseline = random_entry_baseline(oos, n_trades=total_trades)
    print("\n=== Random-entry baseline (same trade count) ===")
    print(random_baseline)

    bars_by_symbol = {}
    for symbol in SYMBOLS:
        raw = read_cached(symbol, "1Min", START, END)
        report = check_quality(raw, symbol)
        bars = exclude_holdout(clean_bars(raw, report))
        bars_by_symbol[symbol] = bars
    bh = buy_and_hold_baseline(bars_by_symbol)
    print(f"\n=== Buy-and-hold baseline (equal-weighted, {START}..{END}) ===")
    print(f"return: {fmt(bh)}")

    regimes = regime_split_metrics(dataset, oos)
    print("\n=== Regime robustness (median realized-volatility split) ===")
    for regime, m in regimes.items():
        print(f"{regime}: {m}")

    # Train the final production model on ALL non-holdout data (not just one fold).
    print("\nTraining final production model on full dataset...")
    final_model = train_fold_model(dataset)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    append_run(
        phase="phase4_model_training",
        data_start=str(START),
        data_end=str(END),
        params={
            "symbols": SYMBOLS,
            "n_folds": N_FOLDS,
            "embargo_days": EMBARGO_DAYS,
            "probability_threshold": PROBABILITY_THRESHOLD,
            "barrier_config": {
                "profit_target_pct": BARRIER_CONFIG.profit_target_pct,
                "stop_loss_pct": BARRIER_CONFIG.stop_loss_pct,
                "max_holding_bars": BARRIER_CONFIG.max_holding_bars,
            },
        },
        win_rate=agg["win_rate"],
        expectancy=agg["expectancy"],
        sharpe=agg["sharpe"],
        max_drawdown=agg["max_drawdown"],
        notes="Phase 4 walk-forward aggregate OOS metrics (signal-level, not the Phase 5 backtester).",
    )
    print("Appended run to experiment_log.csv")

    write_model_card(dataset, balance, fold_metrics, agg, random_baseline, bh, regimes)
    print(f"Wrote model card to {MODEL_CARD_PATH}")

    return 0


def write_model_card(dataset, balance, fold_metrics, agg, random_baseline, bh, regimes) -> None:
    lines = []
    lines.append("# Model Card — Phase 4\n")
    lines.append(
        "Predicts P(triple-barrier profit target hit before stop-loss or timeout) for a long "
        "entry at each regular-session minute bar. Trained on gradient-boosted trees "
        "(`sklearn.ensemble.HistGradientBoostingClassifier`) — see `src/models/train.py` for why "
        "LightGBM wasn't usable on this machine (no libomp, no Homebrew to install it).\n"
    )
    lines.append("## Training data\n")
    lines.append(f"- Symbols: {', '.join(SYMBOLS)}")
    lines.append(f"- Date range: {START} to {END} (regular session only, holdout window excluded)")
    lines.append(f"- Rows after cleaning/labeling: {len(dataset)}")
    lines.append(
        f"- Triple-barrier config: profit_target={BARRIER_CONFIG.profit_target_pct:.1%}, "
        f"stop_loss={BARRIER_CONFIG.stop_loss_pct:.1%}, "
        f"max_holding_bars={BARRIER_CONFIG.max_holding_bars} "
        f"(~{BARRIER_CONFIG.max_holding_bars // 390} regular sessions)"
    )
    lines.append(f"- Features: {', '.join(FEATURE_COLUMNS)}\n")

    edge = agg["expectancy"] - random_baseline["expectancy"]
    edge_pct_of_random = (
        edge / abs(random_baseline["expectancy"]) if random_baseline["expectancy"] else float("nan")
    )
    lines.append("## Key finding\n")
    if abs(edge_pct_of_random) < 0.5:
        lines.append(
            f"**The model does not clearly beat the random-entry baseline at this stage.** "
            f"Aggregate expectancy is {fmt(agg['expectancy'])}/trade vs. "
            f"{fmt(random_baseline['expectancy'])}/trade for random entries at the same trade "
            f"count and risk parameters — a difference of {fmt(edge)} "
            f"({edge_pct_of_random:+.0%} relative to the random baseline). Win rate and Sharpe are "
            "similarly close between the two. This is exactly the baseline-comparison check "
            "docs/pre-mortem.md guard #3 exists to run, and the honest read of it is: this model, "
            "with these features and this universe/date range, isn't demonstrated to add value yet. "
            "**Flagged rather than fixed** — plausible next steps are more/better features, "
            "hyperparameter tuning, a larger training universe, or accepting that a simple long-only "
            "triple-barrier signal on 1-minute bars may not have much edge over these 8 large-caps in "
            "this window. Do not treat downstream Phase 5 backtest results as validating an edge that "
            "wasn't shown here — they inherit this same signal."
        )
    else:
        lines.append(
            f"Model expectancy ({fmt(agg['expectancy'])}/trade) beats the random-entry baseline "
            f"({fmt(random_baseline['expectancy'])}/trade) by {fmt(edge)} "
            f"({edge_pct_of_random:+.0%} relative)."
        )
    lines.append("")

    lines.append("## Class balance (raw triple-barrier label, before binarizing to target)\n")
    lines.append(f"- Counts: {balance['counts']}")
    lines.append(f"- Proportions: {balance['proportions']}")
    lines.append(f"- Flagged skewed (any class < 10%): {balance['is_skewed']}")
    lines.append(
        f"- Dropped for insufficient forward horizon: {balance['n_dropped_insufficient_horizon']}\n"
    )

    lines.append("## Walk-forward validation\n")
    lines.append(f"{N_FOLDS} folds, {EMBARGO_DAYS}-trading-day purge/embargo, expanding-window "
                  f"train. Probability threshold: {PROBABILITY_THRESHOLD}.\n")
    lines.append(
        "| fold | train | test | n_train | n_test | n_trades | accuracy | win_rate | "
        "expectancy | sharpe | max_dd |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for fm in fold_metrics:
        lines.append(
            f"| {fm.fold_id} | {fm.train_start}..{fm.train_end} | {fm.test_start}..{fm.test_end} "
            f"| {fm.n_train} | {fm.n_test} | {fm.n_trades} | {fmt(fm.accuracy)} | "
            f"{fmt(fm.win_rate)} | {fmt(fm.expectancy)} | {fmt(fm.sharpe)} | {fmt(fm.max_drawdown)} |"
        )
    lines.append("")

    lines.append("## Aggregate out-of-sample vs. baselines\n")
    lines.append("These are signal-level metrics from triple-barrier `realized_return` (a "
                  "lightweight proxy backtest, not the cost-aware Phase 5 engine — that's the "
                  "authoritative number, this is for quick model comparison during development).\n")
    lines.append("| | n_trades | win_rate | expectancy | sharpe | max_drawdown |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(
        f"| **Model** | {agg['n_trades']} | {fmt(agg['win_rate'])} | {fmt(agg['expectancy'])} "
        f"| {fmt(agg['sharpe'])} | {fmt(agg['max_drawdown'])} |"
    )
    lines.append(
        f"| Random entry (same trade count) | {random_baseline['n_trades']} | "
        f"{fmt(random_baseline['win_rate'])} | {fmt(random_baseline['expectancy'])} | "
        f"{fmt(random_baseline['sharpe'])} | {fmt(random_baseline['max_drawdown'])} |"
    )
    lines.append(f"| Buy-and-hold (equal-weighted, whole period) | — | — | {fmt(bh)} | — | — |\n")

    lines.append("## Regime robustness (median realized-volatility split by day)\n")
    lines.append("| regime | n_trades | win_rate | expectancy | sharpe | max_drawdown |")
    lines.append("|---|---|---|---|---|---|")
    for regime, m in regimes.items():
        lines.append(
            f"| {regime} | {m['n_trades']} | {fmt(m['win_rate'])} | {fmt(m['expectancy'])} "
            f"| {fmt(m['sharpe'])} | {fmt(m['max_drawdown'])} |"
        )
    lines.append("")

    lines.append("## Known limitations\n")
    lines.append(
        "- Universe and date range are deliberately scoped down for this initial build "
        f"({len(SYMBOLS)} symbols, ~8 months) to keep training/backtesting runtimes tractable "
        "on this machine — widen `SYMBOLS`/`START`/`END` in `scripts/fetch_training_data.py` and "
        "`scripts/train_model.py` for a larger run; nothing else needs to change."
    )
    lines.append(
        "- Regime split is a median realized-volatility split by day, a proxy for "
        "\"trending vs. choppy\" regimes rather than a labeled regime classification."
    )
    lines.append(
        "- These are signal-level metrics (next-bar-agnostic, no costs/slippage). Phase 5's "
        "event-driven backtester is the number that should actually inform a go/no-go decision."
    )

    MODEL_CARD_PATH.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
