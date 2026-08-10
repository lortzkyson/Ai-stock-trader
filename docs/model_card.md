# Model Card — Phase 4

Predicts P(triple-barrier profit target hit before stop-loss or timeout) for a long entry at each regular-session minute bar. Trained on gradient-boosted trees (`sklearn.ensemble.HistGradientBoostingClassifier`) — see `src/models/train.py` for why LightGBM wasn't usable on this machine (no libomp, no Homebrew to install it).

## Training data

- Symbols: AAPL, MSFT, AMZN, GOOGL, NVDA, JPM, WMT, XOM
- Date range: 2025-09-01 to 2026-04-30 (regular session only, holdout window excluded)
- Rows after cleaning/labeling: 508629
- Triple-barrier config: profit_target=2.0%, stop_loss=1.0%, max_holding_bars=1170 (~3 regular sessions)
- Features: ret_5, ret_15, ret_30, ret_60, vol_30, vol_60, rel_vol_30, vwap_dev

## Key finding

**The model does not clearly beat the random-entry baseline at this stage.** Aggregate expectancy is 0.0013/trade vs. 0.0012/trade for random entries at the same trade count and risk parameters — a difference of 0.0001 (+10% relative to the random baseline). Win rate and Sharpe are similarly close between the two. This is exactly the baseline-comparison check docs/pre-mortem.md guard #3 exists to run, and the honest read of it is: this model, with these features and this universe/date range, isn't demonstrated to add value yet. **Flagged rather than fixed** — plausible next steps are more/better features, hyperparameter tuning, a larger training universe, or accepting that a simple long-only triple-barrier signal on 1-minute bars may not have much edge over these 8 large-caps in this window. Do not treat downstream Phase 5 backtest results as validating an edge that wasn't shown here — they inherit this same signal.

## Class balance (raw triple-barrier label, before binarizing to target)

- Counts: {-1.0: 302650, 0.0: 32634, 1.0: 173345}
- Proportions: {-1.0: 0.5950309557654007, 0.0: 0.06416071439103944, 1.0: 0.34080832984355985}
- Flagged skewed (any class < 10%): True
- Dropped for insufficient forward horizon: 0

## Walk-forward validation

5 folds, 4-trading-day purge/embargo, expanding-window train. Probability threshold: 0.5.

| fold | train | test | n_train | n_test | n_trades | accuracy | win_rate | expectancy | sharpe | max_dd |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2025-09-02..2025-10-02 | 2025-10-09..2025-11-14 | 71279 | 84240 | 30942 | 0.5453 | 0.3092 | 0.0006 | 2.3588 | -0.0270 |
| 1 | 2025-09-02..2025-11-10 | 2025-11-17..2025-12-26 | 155519 | 81510 | 35024 | 0.5124 | 0.3262 | 0.0012 | 2.0072 | -0.0262 |
| 2 | 2025-09-02..2025-12-18 | 2025-12-29..2026-02-05 | 237029 | 84240 | 36878 | 0.5010 | 0.3775 | 0.0020 | 4.6361 | -0.0327 |
| 3 | 2025-09-02..2026-01-30 | 2026-02-06..2026-03-17 | 321269 | 84240 | 37077 | 0.5458 | 0.3178 | -0.0002 | -0.9001 | -0.0268 |
| 4 | 2025-09-02..2026-03-11 | 2026-03-18..2026-04-29 | 405509 | 90640 | 35872 | 0.5445 | 0.4223 | 0.0030 | 5.1776 | -0.0161 |

## Aggregate out-of-sample vs. baselines

These are signal-level metrics from triple-barrier `realized_return` (a lightweight proxy backtest, not the cost-aware Phase 5 engine — that's the authoritative number, this is for quick model comparison during development).

| | n_trades | win_rate | expectancy | sharpe | max_drawdown |
|---|---|---|---|---|---|
| **Model** | 175793 | 0.3518 | 0.0013 | 2.7969 | -0.0403 |
| Random entry (same trade count) | 175793 | 0.3472 | 0.0012 | 2.8855 | -0.0557 |
| Buy-and-hold (equal-weighted, whole period) | — | — | 0.2360 | — | — |

## Regime robustness (median realized-volatility split by day)

| regime | n_trades | win_rate | expectancy | sharpe | max_drawdown |
|---|---|---|---|---|---|
| high_vol | 102486 | 0.3517 | 0.0010 | 2.2801 | -0.0266 |
| low_vol | 73307 | 0.3519 | 0.0018 | 3.4770 | -0.0373 |

## Known limitations

- Universe and date range are deliberately scoped down for this initial build (8 symbols, ~8 months) to keep training/backtesting runtimes tractable on this machine — widen `SYMBOLS`/`START`/`END` in `scripts/fetch_training_data.py` and `scripts/train_model.py` for a larger run; nothing else needs to change.
- Regime split is a median realized-volatility split by day, a proxy for "trending vs. choppy" regimes rather than a labeled regime classification.
- These are signal-level metrics (next-bar-agnostic, no costs/slippage). Phase 5's event-driven backtester is the number that should actually inform a go/no-go decision.
