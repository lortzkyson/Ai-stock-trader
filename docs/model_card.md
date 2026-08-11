# Model Card — Phase 4

Predicts P(triple-barrier profit target hit before stop-loss or timeout) for a long entry at each regular-session minute bar. Trained on gradient-boosted trees (`sklearn.ensemble.HistGradientBoostingClassifier`) — see `src/models/train.py` for why LightGBM wasn't usable on this machine (no libomp, no Homebrew to install it).

## Training data

- Symbols: AAPL, MSFT, AMZN, GOOGL, META, NVDA, TSLA, JPM, V, WMT, UNH, XOM, JNJ, PG, HD, MA, BAC, DIS, NFLX, AMD, INTC, CSCO, PFE, KO, PEP, CRM, ADBE, QCOM, COST, ORCL
- Date range: 2025-03-01 to 2026-04-30 (regular session only, holdout window excluded)
- Rows after cleaning/labeling: 3247750
- Triple-barrier config: profit_target=2.0%, stop_loss=1.0%, max_holding_bars=1170 (~3 regular sessions)
- Features: ret_5, ret_15, ret_30, ret_60, vol_30, vol_60, rel_vol_30, vwap_dev, minutes_since_open, rsi_14, range_position_20

## Key finding

**The model does not clearly beat the random-entry baseline at this stage.** Aggregate expectancy is 0.0008/trade vs. 0.0009/trade for random entries at the same trade count and risk parameters — a difference of -0.0001 (-7% relative to the random baseline). Win rate and Sharpe are similarly close between the two. This is exactly the baseline-comparison check docs/pre-mortem.md guard #3 exists to run, and the honest read of it is: this model, with these features and this universe/date range, isn't demonstrated to add value yet. **Flagged rather than fixed** — plausible next steps are more/better features, hyperparameter tuning, a larger training universe, or accepting that a simple long-only triple-barrier signal on 1-minute bars may not have much edge over these 8 large-caps in this window. Do not treat downstream Phase 5 backtest results as validating an edge that wasn't shown here — they inherit this same signal.

## Class balance (raw triple-barrier label, before binarizing to target)

- Counts: {-1.0: 1962823, 0.0: 241993, 1.0: 1042934}
- Proportions: {-1.0: 0.6043639442691093, 0.0: 0.07451096913247633, 1.0: 0.32112508659841427}
- Flagged skewed (any class < 10%): True
- Dropped for insufficient forward horizon: 0

## Walk-forward validation

5 folds, 4-trading-day purge/embargo, expanding-window train. Probability threshold: 0.5.

| fold | train | test | n_train | n_test | n_trades | accuracy | win_rate | expectancy | sharpe | max_dd |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2025-03-03..2025-05-02 | 2025-05-09..2025-07-18 | 497586 | 513753 | 149115 | 0.5823 | 0.3539 | 0.0021 | 5.4183 | -0.0308 |
| 1 | 2025-03-03..2025-07-14 | 2025-07-21..2025-09-25 | 1012108 | 517668 | 235438 | 0.5401 | 0.3407 | 0.0015 | 5.0645 | -0.0187 |
| 2 | 2025-03-03..2025-09-19 | 2025-09-26..2025-12-03 | 1529398 | 529461 | 332387 | 0.4704 | 0.3267 | 0.0006 | 1.8706 | -0.0316 |
| 3 | 2025-03-03..2025-11-26 | 2025-12-04..2026-02-12 | 2067021 | 549447 | 360293 | 0.4618 | 0.3382 | 0.0009 | 2.1673 | -0.0246 |
| 4 | 2025-03-03..2026-02-06 | 2026-02-13..2026-04-29 | 2605933 | 595019 | 399453 | 0.4413 | 0.3218 | 0.0001 | 1.1803 | -0.0569 |

## Aggregate out-of-sample vs. baselines

These are signal-level metrics from triple-barrier `realized_return` (a lightweight proxy backtest, not the cost-aware Phase 5 engine — that's the authoritative number, this is for quick model comparison during development).

| | n_trades | win_rate | expectancy | sharpe | max_drawdown |
|---|---|---|---|---|---|
| **Model** | 1476686 | 0.3332 | 0.0008 | 3.1113 | -0.0596 |
| Random entry (same trade count) | 1476686 | 0.3212 | 0.0009 | 2.9765 | -0.0582 |
| Buy-and-hold (equal-weighted, whole period) | — | — | 0.2728 | — | — |

## Regime robustness (median realized-volatility split by day)

| regime | n_trades | win_rate | expectancy | sharpe | max_drawdown |
|---|---|---|---|---|---|
| high_vol | 787782 | 0.3265 | 0.0005 | 1.8795 | -0.0543 |
| low_vol | 688904 | 0.3408 | 0.0012 | 4.0244 | -0.0297 |

## Known limitations

- Universe/date range (30 symbols, 2025-03-01 to 2026-04-30) is still a scope-down from the full ~10-year history Alpaca actually has available — widen `SYMBOLS`/`START`/`END` in `scripts/fetch_training_data.py` and `scripts/train_model.py` further if needed, but note that widening from the original 8 symbols/~8 months to this run's 30 symbols/~14 months (plus 3 added features: `minutes_since_open`, `rsi_14`, `range_position_20`) did not produce a clearer edge over the random baseline — see the Key Finding above.
- Regime split is a median realized-volatility split by day, a proxy for "trending vs. choppy" regimes rather than a labeled regime classification.
- These are signal-level metrics (next-bar-agnostic, no costs/slippage). Phase 5's event-driven backtester is the number that should actually inform a go/no-go decision.
