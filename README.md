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

## Known limitations

- **Survivorship bias in the ticker universe** (`data/universe.csv`): the seed universe is a manually curated list of today's liquid large-cap names, not a point-in-time index membership feed. It excludes tickers that were delisted, acquired, or went bankrupt during any historical backtest period, so backtests will overstate performance relative to a survivorship-bias-free universe. Getting point-in-time constituent data typically requires a paid reference-data subscription; this is deferred rather than solved. See `src/data/universe.py` and [docs/pre-mortem.md](docs/pre-mortem.md) guard #2.
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
make test        # pytest with coverage
make lint         # ruff
make typecheck    # mypy
make check        # lint + typecheck + test
make data-quality  # flag gaps/dupes/impossible values in cached bar data
make backtest      # run the backtester (from Phase 5 onward)
make paper-trade   # run the live paper-trading loop (from Phase 7 onward)
```

`make backtest` and `make paper-trade` are placeholders until the corresponding phases build `src/backtest/run.py` and `src/execution/run.py`.

## Experiment log and holdout

Every training/backtest run gets appended to `experiment_log.csv` (git-tracked, append-only) once Phase 4 introduces it. A final holdout period is reserved and untouched until Phase 8 — see [docs/pre-mortem.md §5](docs/pre-mortem.md#5-experiment-logging-and-the-final-holdout) before touching data pulled by Phase 2.

## Status

- [x] Phase 0 — pre-mortem / design spec
- [x] Phase 1 — project scaffold
- [x] Phase 2 — data pipeline
- [x] Phase 3 — labeling and features
- [x] Phase 4 — model training
- [x] Phase 5 — backtesting engine (this commit)
- [x] Phase 6 — risk management layer (built before Phase 5 since the backtester depends on it)
- [ ] Phase 7 — paper trading integration
- [ ] Phase 8 — go/no-go review before real money
- [ ] Phase 9 — ongoing monitoring & retraining
