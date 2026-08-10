# ai-stock-trader

Day-trading AI system, built in phases. Start with [docs/pre-mortem.md](docs/pre-mortem.md) — it's the living design spec every phase below follows. Key constraints from that doc: starting equity is under $25k (PDT rule applies), the strategy is a hybrid of day trades and occasional multi-day swing holds, and the target metric is **expectancy per trade + Sharpe/Sortino**, not raw win rate.

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

## Setup

```bash
make install
cp .env.example .env   # then fill in your Alpaca + market-data keys
```

## Common commands

```bash
make test        # pytest with coverage
make lint         # ruff
make typecheck    # mypy
make check        # lint + typecheck + test
make backtest      # run the backtester (from Phase 5 onward)
make paper-trade   # run the live paper-trading loop (from Phase 7 onward)
```

`make backtest` and `make paper-trade` are placeholders until the corresponding phases build `src/backtest/run.py` and `src/execution/run.py`.

## Experiment log and holdout

Every training/backtest run gets appended to `experiment_log.csv` (git-tracked, append-only) once Phase 4 introduces it. A final holdout period is reserved and untouched until Phase 8 — see [docs/pre-mortem.md §5](docs/pre-mortem.md#5-experiment-logging-and-the-final-holdout) before touching data pulled by Phase 2.

## Status

- [x] Phase 0 — pre-mortem / design spec
- [x] Phase 1 — project scaffold (this commit)
- [ ] Phase 2 — data pipeline
- [ ] Phase 3 — labeling and features
- [ ] Phase 4 — model training
- [ ] Phase 5 — backtesting engine
- [ ] Phase 6 — risk management layer
- [ ] Phase 7 — paper trading integration
- [ ] Phase 8 — go/no-go review before real money
- [ ] Phase 9 — ongoing monitoring & retraining
