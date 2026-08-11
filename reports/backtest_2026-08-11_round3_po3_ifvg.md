# Backtest Report — Phase 5 (2026-08-11)

Event-driven backtest replaying Phase 4's model signals against real historical bars for the same 30 symbols, using the exact out-of-sample walk-forward test periods from Phase 4 (no re-tuning here) — see `docs/model_card.md` for the model itself.

## Configuration

- Starting equity: $10,000 (deliberately under $25k — exercises the PDT path)
- Order type: market, 5 bps slippage, no added latency beyond mandatory next-bar-open
- Fees: Alpaca's actual schedule (`src/backtest/costs.py`) — $0 commission, SEC fee + FINRA TAF on sells
- Risk: 1.0% risked per trade (fixed-fractional), 10% max-drawdown circuit breaker, 2% daily loss limit

## Results

| metric | value |
|---|---|
| n_trades | 301 |
| win_rate | 0.3688 |
| expectancy | -0.0008 |
| profit_factor | 0.8571 |
| sharpe | -0.8305 |
| sortino | -0.7675 |
| max_drawdown | -0.0714 |
| avg_trade_duration_bars | 161.2159 |
| total_return | -0.0479 |

## Statistical significance and concentration

A backtest can look profitable purely by chance with a small trade count, or look profitable in aggregate while really being one or two lucky symbols masking a loss everywhere else. Both are checked here before the headline numbers above get any credit as a demonstrated edge.

- One-sample t-test on per-trade return vs. zero: t=-1.151, p=0.251
  -> **not statistically significant** — consistent with pure chance
- Profit concentration: the single best-performing symbol accounts for 32% of total gross positive P&L
- 8 of 26 traded symbols were net positive

**Verdict: this result should not be read as a demonstrated edge.** With only 301 real (non-overlapping) trades and a p-value well above 0.05, the positive headline numbers above are within the range pure noise would produce. This is consistent with Phase 4's much larger signal-level sample (1.48M predictions), which showed no edge over a random baseline — that larger, more reliable sample should be trusted over this smaller number of actual position-limited trades.

## Exit reasons

- stop_loss: 175
- take_profit: 68
- max_holding: 58

Day trades: 216 (71.8% of all trades)

## In-sample vs. out-of-sample (overfitting check)

Averaged across folds, at the model-signal level (not full backtest fills).

| | in-sample (train) | out-of-sample (test) |
|---|---|---|
| win_rate | 0.2991 | 0.2554 |
| expectancy | 0.0014 | 0.0009 |
| sharpe | 3.7688 | 2.7147 |
| max_drawdown | -0.0360 | -0.0264 |

No significant in-sample/out-of-sample divergence detected by the threshold used here — not the same as proof the model generalizes, just that this particular overfitting signature isn't present.

## Known limitations

- Same universe/date-range as Phase 4 (30 symbols, 2025-03-01 to 2026-04-30) — see `docs/model_card.md`.
- Exits (stop-loss/take-profit/max-holding) fill as market orders; only entries model the market-vs-limit distinction (`src/backtest/fills.py`).
- PDT handling blocks *all* new entries once the day-trade budget is spent, not just same-day ones — a conservative simplification, documented in `src/backtest/engine.py`.
- **Phase 4's own model card already flags that this model doesn't clearly beat a random-entry baseline.** These backtest numbers inherit that same signal — a profitable-looking backtest here would not by itself contradict that finding, since both use the same underlying predictions.
