# Runbook

**Current status: paper-trading infrastructure only, nothing running unattended, real money not yet approved — see [docs/go_live_review.md](go_live_review.md) for the current NO-GO and why.** This runbook describes the maintenance discipline to run once that changes, and what's already usable today (`make monitor`, the kill switch) regardless.

## What to check on a normal day

```bash
make monitor
```

Shows account equity, open positions, and today's day-trade count against the PDT limit (3 per rolling 5 business days under $25k equity). Also worth a periodic look:

- `data/execution_log.jsonl` — structured log of every signal, order, fill, and reconciliation mismatch. Grep for `"event_type": "reconciliation_mismatch"` — any hit there means local state and Alpaca's actual account disagreed and should be understood before trusting the system further, not dismissed.
- `.venv/bin/python scripts/kill_switch.py status` — confirm it's `not engaged` unless you deliberately engaged it.

## If the kill switch fires

The kill switch (`scripts/kill_switch.py`) is manual — it only engages because a person (you) ran `engage`, so "it fired" really means "you decided something looked wrong." When it's engaged:

1. No new orders will be submitted (existing open orders/positions are untouched unless you also passed `--flatten`).
2. Investigate before disengaging — check `data/execution_log.jsonl` for what was happening right before you engaged it, check `make monitor` for anything unexpected in positions/equity, check whether a circuit breaker (below) also tripped around the same time.
3. Only run `scripts/kill_switch.py disengage` once you understand what triggered your concern. Re-engaging blind because "it's probably fine" defeats the point of having it.

## If an automatic circuit breaker trips

Two independent breakers live in `src/risk/engine.py` (`RiskEngine.daily_loss_limit`, `RiskEngine.drawdown_breaker`):

- **Daily loss limit** (default 2% of the day's starting equity): blocks new entries for the rest of that trading day, then **resets automatically** at the start of the next day. No action needed unless it's tripping repeatedly across multiple days — that's a signal to look at §"Retraining triggers" below, not just wait it out.
- **Max drawdown breaker** (default 10% from peak equity): blocks new entries and **does not auto-reset** — by design, this needs a human to call `manual_reset()` after actually reviewing what happened. Don't reset it reflexively; review the trade log first, understand whether it was a bad stretch of variance or a sign the model/market relationship has broken down, and only reset once you'd be comfortable explaining the drawdown to someone else.

Both are unit-tested (`tests/risk/test_engine.py`) with a simulated losing streak confirming they actually halt trading, but as of this writing neither has been exercised against a real live/paper order flow — see docs/go_live_review.md §3. The first real trip is worth treating as a live-integration test in its own right, not just routine.

## Retraining

**Cadence:** retrain monthly by default.

**Trigger conditions for an off-cycle retrain** (any of):
- Live/paper expectancy stays below zero for 10 consecutive trading days.
- `scripts/generate_performance_report.py` flags divergence from the backtest baseline on more than one metric in the same run.
- A material change to the universe, timeframe, or risk parameters.

**Every retrain — scheduled or triggered — goes through the same process, no exceptions for "it's just a refresh":**

```bash
make train      # scripts/fetch_training_data.py + scripts/train_model.py
make backtest    # scripts/run_backtest.py, over Phase 4's walk-forward OOS periods
```

This means: walk-forward validation with purge/embargo (not a quick in-sample check), a fresh entry in `experiment_log.csv`, an updated `docs/model_card.md`, and an updated backtest report — the exact discipline from Phases 4-5, every time. The temptation to skip straight to "just bump the training window and ship it" is exactly the iteration-without-logging failure mode docs/pre-mortem.md §5 exists to prevent.

The final holdout period (`2026-05-01` through `2026-07-31`, pinned in `src/data/holdout.py`) stays untouched on every retrain until Phase 8 is ready to be revisited for real — retraining doesn't get to peek at it either.

## Performance monitoring

```bash
.venv/bin/python scripts/generate_performance_report.py
```

Reconstructs round-trip trades from `data/execution_log.jsonl`'s fill events and compares win rate / expectancy / Sharpe / max drawdown against the last backtest run, flagging any metric that diverges by more than 50% relative. Run this on the same cadence as your paper/live trading loop (daily is reasonable once one exists). Right now, with no fills logged yet, it correctly reports nothing to compare.
