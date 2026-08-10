#!/usr/bin/env python3
"""Phase 5: run the realistic event-driven backtester over the exact
out-of-sample walk-forward periods from Phase 4 (same folds, same config —
not a new tuning run), and check in-sample vs out-of-sample divergence as an
overfitting tell.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from backtest.engine import run_backtest  # noqa: E402
from backtest.fills import FillConfig  # noqa: E402
from backtest.metrics import compute_backtest_metrics  # noqa: E402
from features.labeling import TripleBarrierConfig  # noqa: E402
from models.dataset import build_dataset  # noqa: E402
from models.experiment_log import append_run  # noqa: E402
from models.metrics import compute_signal_metrics  # noqa: E402
from models.train import train_fold_model  # noqa: E402
from models.validation import generate_walk_forward_folds, split_fold  # noqa: E402
from risk.engine import RiskConfig, RiskEngine  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SYMBOLS = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "JPM", "WMT", "XOM"]
START = date(2025, 9, 1)
END = date(2026, 4, 30)
BARRIER_CONFIG = TripleBarrierConfig()
N_FOLDS = 5
EMBARGO_DAYS = 4
PROBABILITY_THRESHOLD = 0.5
STARTING_EQUITY = 10_000.0  # under $25k -> exercises the PDT path deliberately

REPORT_PATH = REPO_ROOT / "reports" / f"backtest_{datetime.now(timezone.utc):%Y-%m-%d}.md"


def fmt(x: float) -> str:
    return "nan" if x != x else f"{x:.4f}"  # noqa: PLR0124


def main() -> int:
    print(f"Building dataset: {SYMBOLS} {START}..{END}")
    dataset = build_dataset(SYMBOLS, START, END, barrier_config=BARRIER_CONFIG)

    trading_dates = sorted(dataset["date"].unique())
    folds = generate_walk_forward_folds(trading_dates, n_folds=N_FOLDS, embargo_days=EMBARGO_DAYS)

    oos_frames = []
    in_sample_metrics_per_fold = []
    out_sample_metrics_per_fold = []

    for fold in folds:
        train_df, test_df = split_fold(dataset, fold)
        if len(train_df) == 0 or len(test_df) == 0:
            continue

        model = train_fold_model(train_df)

        train_proba = model.predict_proba(train_df[_feature_columns()])[:, 1]
        train_pred = (train_proba >= PROBABILITY_THRESHOLD).astype(int)
        in_sample_traded = train_df.loc[train_pred == 1]
        in_sample_metrics_per_fold.append(
            compute_signal_metrics(
                in_sample_traded["realized_return"].to_numpy(),
                in_sample_traded["label"].to_numpy(),
                in_sample_traded["date"].to_numpy(),
            )
        )

        test_proba = model.predict_proba(test_df[_feature_columns()])[:, 1]
        test_pred = (test_proba >= PROBABILITY_THRESHOLD).astype(int)
        out_sample_traded = test_df.loc[test_pred == 1]
        out_sample_metrics_per_fold.append(
            compute_signal_metrics(
                out_sample_traded["realized_return"].to_numpy(),
                out_sample_traded["label"].to_numpy(),
                out_sample_traded["date"].to_numpy(),
            )
        )

        oos = test_df[["timestamp", "symbol", "open", "high", "low", "close", "volume"]].copy()
        oos["predicted"] = test_pred
        oos_frames.append(oos)

    oos_all = pd.concat(oos_frames, ignore_index=True)
    print(f"OOS bar-rows spanning all folds: {len(oos_all)}")

    bars_by_symbol: dict[str, pd.DataFrame] = {}
    signals_by_symbol: dict[str, pd.Series] = {}
    for symbol, group in oos_all.groupby("symbol"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        bars_by_symbol[symbol] = group[["timestamp", "open", "high", "low", "close", "volume"]]
        signals_by_symbol[symbol] = group["predicted"]

    trading_calendar = sorted({pd.Timestamp(ts).date() for ts in oos_all["timestamp"]})

    risk_config = RiskConfig(barrier_config=BARRIER_CONFIG)
    risk_engine = RiskEngine(risk_config)
    fill_config = FillConfig(order_type="market", slippage_bps=5.0, latency_bars=0)

    print(f"Running backtest: starting_equity=${STARTING_EQUITY:,.0f}, {len(bars_by_symbol)} symbols")
    result = run_backtest(
        bars_by_symbol, signals_by_symbol, risk_engine, fill_config,
        starting_equity=STARTING_EQUITY, trading_calendar=trading_calendar,
    )
    print(f"Trades executed: {len(result.trades)}")

    metrics = compute_backtest_metrics(result.trades, result.daily_equity)
    print("Backtest metrics:", metrics)

    in_sample_agg = _average_metrics(in_sample_metrics_per_fold)
    out_sample_agg = _average_metrics(out_sample_metrics_per_fold)
    divergence_flag = _check_divergence(in_sample_agg, out_sample_agg)

    append_run(
        phase="phase5_backtest",
        data_start=str(START),
        data_end=str(END),
        params={
            "symbols": SYMBOLS,
            "n_folds": N_FOLDS,
            "embargo_days": EMBARGO_DAYS,
            "starting_equity": STARTING_EQUITY,
            "fill_config": {
                "order_type": fill_config.order_type,
                "slippage_bps": fill_config.slippage_bps,
                "latency_bars": fill_config.latency_bars,
            },
            "risk_config": {
                "risk_per_trade_pct": risk_config.risk_per_trade_pct,
                "max_position_fraction": risk_config.max_position_fraction,
                "daily_loss_limit_pct": risk_config.daily_loss_limit_pct,
                "max_drawdown_pct": risk_config.max_drawdown_pct,
                "pdt_equity_threshold": risk_config.pdt_equity_threshold,
            },
        },
        win_rate=metrics["win_rate"],
        expectancy=metrics["expectancy"],
        sharpe=metrics["sharpe"],
        sortino=metrics["sortino"],
        max_drawdown=metrics["max_drawdown"],
        notes="Phase 5 event-driven backtest over Phase 4's exact OOS walk-forward periods.",
    )
    print("Appended run to experiment_log.csv")

    write_report(dataset, result, metrics, in_sample_agg, out_sample_agg, divergence_flag)
    print(f"Wrote report to {REPORT_PATH}")

    return 0


def _feature_columns() -> list[str]:
    from features.engineering import FEATURE_COLUMNS

    return list(FEATURE_COLUMNS)


def _average_metrics(metrics_list: list[dict]) -> dict:
    keys = ["win_rate", "expectancy", "sharpe", "max_drawdown"]
    out = {}
    for k in keys:
        vals = [m[k] for m in metrics_list if m[k] == m[k]]  # drop NaN
        out[k] = sum(vals) / len(vals) if vals else float("nan")
    out["n_trades"] = sum(m["n_trades"] for m in metrics_list)
    return out


def _check_divergence(in_sample: dict, out_sample: dict) -> str | None:
    in_exp, out_exp = in_sample["expectancy"], out_sample["expectancy"]
    if in_exp != in_exp or out_exp != out_exp:
        return None
    gap = in_exp - out_exp
    if in_exp > 0 and gap > 0.5 * abs(in_exp) and gap > 0.0005:
        return (
            f"In-sample expectancy ({fmt(in_exp)}/trade) is notably higher than "
            f"out-of-sample ({fmt(out_exp)}/trade) — a classic overfitting tell. "
            "The model fits its own training data better than it generalizes."
        )
    return None


def write_report(dataset, result, metrics, in_sample_agg, out_sample_agg, divergence_flag) -> None:
    lines = [f"# Backtest Report — Phase 5 ({datetime.now(timezone.utc):%Y-%m-%d})\n"]
    lines.append(
        "Event-driven backtest replaying Phase 4's model signals against real historical bars "
        "for the same 8 symbols, using the exact out-of-sample walk-forward test periods from "
        "Phase 4 (no re-tuning here) — see `docs/model_card.md` for the model itself.\n"
    )

    lines.append("## Configuration\n")
    lines.append(f"- Starting equity: ${STARTING_EQUITY:,.0f} (deliberately under $25k — exercises the PDT path)")
    lines.append("- Order type: market, 5 bps slippage, no added latency beyond mandatory next-bar-open")
    lines.append("- Fees: Alpaca's actual schedule (`src/backtest/costs.py`) — $0 commission, SEC fee + FINRA TAF on sells")
    lines.append(
        f"- Risk: {RiskConfig().risk_per_trade_pct:.1%} risked per trade (fixed-fractional), "
        f"{RiskConfig().max_drawdown_pct:.0%} max-drawdown circuit breaker, "
        f"{RiskConfig().daily_loss_limit_pct:.0%} daily loss limit\n"
    )

    lines.append("## Results\n")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for k in ["n_trades", "win_rate", "expectancy", "profit_factor", "sharpe", "sortino",
              "max_drawdown", "avg_trade_duration_bars", "total_return"]:
        lines.append(f"| {k} | {fmt(metrics[k]) if isinstance(metrics[k], float) else metrics[k]} |")
    lines.append("")

    if not result.trades.empty:
        lines.append("## Exit reasons\n")
        counts = result.trades["exit_reason"].value_counts()
        for reason, count in counts.items():
            lines.append(f"- {reason}: {count}")
        lines.append("")
        day_trade_frac = result.trades["is_day_trade"].mean()
        lines.append(f"Day trades: {int(result.trades['is_day_trade'].sum())} "
                      f"({day_trade_frac:.1%} of all trades)\n")

    lines.append("## In-sample vs. out-of-sample (overfitting check)\n")
    lines.append("Averaged across folds, at the model-signal level (not full backtest fills).\n")
    lines.append("| | in-sample (train) | out-of-sample (test) |")
    lines.append("|---|---|---|")
    for k in ["win_rate", "expectancy", "sharpe", "max_drawdown"]:
        lines.append(f"| {k} | {fmt(in_sample_agg[k])} | {fmt(out_sample_agg[k])} |")
    lines.append("")
    if divergence_flag:
        lines.append(f"**Flagged:** {divergence_flag}\n")
    else:
        lines.append(
            "No significant in-sample/out-of-sample divergence detected by the threshold used "
            "here — not the same as proof the model generalizes, just that this particular "
            "overfitting signature isn't present.\n"
        )

    lines.append("## Known limitations\n")
    lines.append(
        "- Same universe/date-range scope-down as Phase 4 (8 symbols, ~8 months) — see "
        "`docs/model_card.md`."
    )
    lines.append(
        "- Exits (stop-loss/take-profit/max-holding) fill as market orders; only entries model "
        "the market-vs-limit distinction (`src/backtest/fills.py`)."
    )
    lines.append(
        "- PDT handling blocks *all* new entries once the day-trade budget is spent, not just "
        "same-day ones — a conservative simplification, documented in `src/backtest/engine.py`."
    )
    lines.append(
        "- **Phase 4's own model card already flags that this model doesn't clearly beat a "
        "random-entry baseline.** These backtest numbers inherit that same signal — a "
        "profitable-looking backtest here would not by itself contradict that finding, since "
        "both use the same underlying predictions."
    )

    REPORT_PATH.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
