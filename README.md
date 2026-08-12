# ai-stock-trader

Day-trading AI system, built in phases. Start with [docs/pre-mortem.md](docs/pre-mortem.md) — it's the living design spec every phase below follows. Key constraints from that doc: starting equity is under $25k (PDT rule applies), the strategy is a hybrid of day trades and occasional multi-day swing holds, and the target metric is **expectancy per trade + Sharpe/Sortino**, not raw win rate.

> **Current status: NO-GO on real money.** All 9 phases are built and tested. Four strategy attempts (three intraday ML variants, one cross-sectional momentum) have found no edge that survives scrutiny. Survivorship bias is now fixed — and fixing it collapsed the most promising result from +2,140% to +249%, turning an apparent SPY-beater into a SPY-underperformer. See [docs/go_live_review.md](docs/go_live_review.md) and [docs/next_steps.md](docs/next_steps.md). No live-order-submission code exists anywhere in this repo.

## Architecture

Each `src/` package corresponds to one build phase and one concern. The backtester and the live executor both call into `risk/` and `features/` rather than each having their own copy — that's deliberate (see pre-mortem, guard #9 and the Phase 7 parity test).

| Package | Responsibility |
|---|---|
| `src/data/` | Historical + live market data ingestion, caching (parquet), data-quality checks, universe/liquidity filtering. |
| `src/features/` | Lookahead-safe feature engineering and triple-barrier labeling. Shared by backtest and live paths. |
| `src/models/` | Model training with walk-forward validation, experiment logging, model cards. |
| `src/backtest/` | Event-driven backtester: realistic fills, costs, slippage, order-type modeling. |
| `src/risk/` | Standalone risk engine: position sizing, stops, daily loss limit, drawdown circuit breaker, PDT day-trade tracking. Used by both backtest and execution. |
| `src/execution/` | Alpaca order submission, reconciliation, kill switch. Paper first, live only after the Phase 8 go/no-go review. |
| `src/monitoring/` | Live dashboards/reports, performance-vs-backtest divergence checks. |

`tests/` mirrors `src/` package-for-package.

## Known limitations

- **Survivorship bias — RESOLVED for daily/cross-sectional strategies** (2026-08-12). Alpaca's asset list is current-only, but it *will* serve history for delisted tickers if you know the symbol, so the missing list is recoverable by brute force: `scripts/discover_universe.py` sweeps the full 1-4 letter ticker space at 21 historical dates. Recovery of known delistings went from 2/16 to 16/16; the universe grew 2,150 → 5,079. This mattered enormously — see [docs/next_steps.md](docs/next_steps.md), where correcting it collapsed a momentum backtest from +2,140% to +249%. Note the older minute-bar work (`data/universe.csv`, Phases 2-5) still uses the survivor-only large-cap list and remains biased.
- **Halt/illiquidity detection is a proxy**, not a labeled feed: `src/data/quality.py` flags any trading day missing more than 5 minutes of expected minute bars as "halt-like" and excludes it from cleaned output. This catches real halts and severe illiquidity but could also flag a legitimately quiet session; the reasoning is documented in that module's docstring.
- **Extended-hours bars are dropped.** Alpaca's default response includes pre/post-market bars; `clean_bars` restricts to the regular 09:30-16:00 ET session, since extended-hours volume is too thin for realistic fills (pre-mortem guard #6) and none of the phases as scoped trade outside regular hours. Confirmed against a real pull that this — and exchange-local vs. UTC timestamp handling — actually mattered: naively treating Alpaca's UTC timestamps as already exchange-local caused every trading day to misfire as "halt-like." Fixed in `src/data/quality.py`, covered by regression tests.

## Data feed

Historical market data comes from Alpaca (`src/data/alpaca_client.py`), using the full consolidated SIP feed at no cost — see [docs/data_feed_decision.md](docs/data_feed_decision.md) for why the free plan's real-time-only IEX restriction doesn't apply to historical/backtest queries, and what that means for live trading in Phase 7+.

## Setup

```bash
make install
cp .env.example .env   # then fill in your Alpaca API key/secret
```

## Common commands

```bash
make test         # pytest with coverage
make lint          # ruff
make typecheck     # mypy
make check         # lint + typecheck + test
make data-quality   # flag gaps/dupes/impossible values in cached bar data
make train           # pull training data + walk-forward train + write model card
make backtest         # run the Phase 5 event-driven backtest over Phase 4's OOS periods
make monitor           # print live account/positions/PDT-count dashboard (paper)

# Runs one iteration and exits (kill switch/market-clock check, exits, entries, state persistence):
.venv/bin/python scripts/paper_trading_loop.py

# Kill switch takes an action argument, so call it directly:
.venv/bin/python scripts/kill_switch.py {status,engage,disengage} [--flatten]
```

`scripts/paper_trading_loop.py` is complete, verified against the real Alpaca paper endpoint, and **currently scheduled** via a `launchd` agent (every 5 minutes, whenever the Mac is awake) — see [docs/runbook.md](docs/runbook.md) for what each iteration does and how to inspect or stop it. It trades paper money only; no live-order path exists. Bear in mind it's running the Phase 3-5 intraday model, which has no demonstrated edge — treat it as an operational track record, not strategy validation.

## Experiment log and holdout

Every training/backtest run gets appended to `experiment_log.csv` (git-tracked, append-only) once Phase 4 introduces it. A final holdout period is reserved and untouched until Phase 8 — see [docs/pre-mortem.md §5](docs/pre-mortem.md#5-experiment-logging-and-the-final-holdout) before touching data pulled by Phase 2.

## Status

- [x] Phase 0 — pre-mortem / design spec
- [x] Phase 1 — project scaffold
- [x] Phase 2 — data pipeline
- [x] Phase 3 — labeling and features
- [x] Phase 4 — model training
- [x] Phase 5 — backtesting engine
- [x] Phase 6 — risk management layer (built before Phase 5 since the backtester depends on it)
- [x] Phase 7 — paper trading integration
- [x] Phase 8 — go/no-go review before real money: **[NO-GO](docs/go_live_review.md)** — no live-trading code exists, none should until this changes
- [x] Phase 9 — ongoing monitoring & retraining (this commit)
