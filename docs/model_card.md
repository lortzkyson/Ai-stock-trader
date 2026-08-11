# Model Card — Phase 4

Predicts P(triple-barrier profit target hit before stop-loss or timeout) for a long entry at each regular-session minute bar. Trained on gradient-boosted trees (`sklearn.ensemble.HistGradientBoostingClassifier`) — see `src/models/train.py` for why LightGBM wasn't usable on this machine (no libomp, no Homebrew to install it).

## Training data

- Symbols: AAPL, MSFT, AMZN, GOOGL, META, NVDA, TSLA, JPM, V, WMT, UNH, XOM, JNJ, PG, HD, MA, BAC, DIS, NFLX, AMD, INTC, CSCO, PFE, KO, PEP, CRM, ADBE, QCOM, COST, ORCL
- Date range: 2025-03-01 to 2026-04-30 (regular session only, holdout window excluded)
- Rows after cleaning/labeling: 3249322
- Triple-barrier config: profit_target=2.0%, stop_loss=1.0%, max_holding_bars=390 (~1 regular sessions)
- Features: ret_5, ret_15, ret_30, ret_60, vol_30, vol_60, rel_vol_30, vwap_dev, minutes_since_open, rsi_14, range_position_20, dist_to_accum_high, dist_to_accum_low, swept_accum_high_recent, swept_accum_low_recent, bars_since_sweep, sweep_direction, bull_fvg_active, bull_fvg_dist, bear_fvg_active, bear_fvg_dist, bars_since_bull_ifvg, bars_since_bear_ifvg

## Key finding

**The model does not clearly beat the random-entry baseline at this stage.** Aggregate expectancy is 0.0006/trade vs. 0.0007/trade for random entries at the same trade count and risk parameters — a difference of -0.0001 (-9% relative to the random baseline). Win rate and Sharpe are similarly close between the two. This is exactly the baseline-comparison check docs/pre-mortem.md guard #3 exists to run, and the honest read of it is: this model, with these features and this universe/date range, isn't demonstrated to add value yet. **Flagged rather than fixed** — plausible next steps are more/better features, hyperparameter tuning, a larger training universe, or accepting that a simple long-only triple-barrier signal on 1-minute bars may not have much edge over these 8 large-caps in this window. Do not treat downstream Phase 5 backtest results as validating an edge that wasn't shown here — they inherit this same signal.

## Class balance (raw triple-barrier label, before binarizing to target)

- Counts: {-1.0: 1501307, 0.0: 1131628, 1.0: 616387}
- Proportions: {-1.0: 0.46203700341178866, 0.0: 0.3482658843906513, 1.0: 0.18969711219755997}
- Flagged skewed (any class < 10%): False
- Dropped for insufficient forward horizon: 0

## Walk-forward validation

5 folds, 4-trading-day purge/embargo, expanding-window train. Probability threshold: 0.5.

| fold | train | test | n_train | n_test | n_trades | accuracy | win_rate | expectancy | sharpe | max_dd |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2025-03-03..2025-05-02 | 2025-05-09..2025-07-18 | 497586 | 513753 | 90638 | 0.7500 | 0.2771 | 0.0020 | 4.8120 | -0.0159 |
| 1 | 2025-03-03..2025-07-14 | 2025-07-21..2025-09-25 | 1012108 | 517668 | 167812 | 0.6816 | 0.2486 | 0.0010 | 3.2487 | -0.0184 |
| 2 | 2025-03-03..2025-09-19 | 2025-09-26..2025-12-03 | 1529398 | 529461 | 264152 | 0.5607 | 0.2406 | 0.0002 | 2.0862 | -0.0360 |
| 3 | 2025-03-03..2025-11-26 | 2025-12-04..2026-02-12 | 2067021 | 549447 | 248303 | 0.5923 | 0.2539 | 0.0005 | 0.7531 | -0.0228 |
| 4 | 2025-03-03..2026-02-06 | 2026-02-13..2026-04-29 | 2605933 | 596591 | 352772 | 0.4980 | 0.2566 | 0.0006 | 2.6738 | -0.0388 |

## Aggregate out-of-sample vs. baselines

These are signal-level metrics from triple-barrier `realized_return` (a lightweight proxy backtest, not the cost-aware Phase 5 engine — that's the authoritative number, this is for quick model comparison during development).

| | n_trades | win_rate | expectancy | sharpe | max_drawdown |
|---|---|---|---|---|---|
| **Model** | 1123677 | 0.2527 | 0.0006 | 2.8103 | -0.0388 |
| Random entry (same trade count) | 1123677 | 0.1819 | 0.0007 | 2.6448 | -0.0416 |
| Buy-and-hold (equal-weighted, whole period) | — | — | 0.2728 | — | — |

## Regime robustness (median realized-volatility split by day)

| regime | n_trades | win_rate | expectancy | sharpe | max_drawdown |
|---|---|---|---|---|---|
| high_vol | 691033 | 0.2404 | 0.0003 | 1.9189 | -0.0349 |
| low_vol | 432644 | 0.2723 | 0.0012 | 3.4659 | -0.0222 |

## Known limitations

- Universe/date range (30 symbols, 2025-03-01 to 2026-04-30) is still a scope-down from the full ~10-year history Alpaca actually has available — widen `SYMBOLS`/`START`/`END` in `scripts/fetch_training_data.py` and `scripts/train_model.py` further if needed. See docs/go_live_review.md §2 for the full history of attempts (generic features at two different scopes, then this dedicated PO3/IFVG feature set) and why none showed an edge.
- Regime split is a median realized-volatility split by day, a proxy for "trending vs. choppy" regimes rather than a labeled regime classification.
- These are signal-level metrics (next-bar-agnostic, no costs/slippage). Phase 5's event-driven backtester is the number that should actually inform a go/no-go decision.
