# Pre-Mortem: Day-Trading AI System

**Status:** Living design spec. Update this document first if timeframe, risk tolerance, or account size assumptions change — downstream phases follow this doc, not the other way around.

**Context:** Rebuilding from scratch. A previous attempt produced a 33% win rate; no code survives from it, so we can't diagnose what went wrong directly. This document assumes the failure mode is unknown and guards against the *most common* causes rather than a specific one.

**Account assumption (user-provided):** Starting equity is **under $25,000**. This is load-bearing — see §4.

---

## 1. Common failure modes in retail algo day-trading systems

| # | Failure mode | What it looks like in practice |
|---|---|---|
| 1 | Lookahead bias | A feature or label at time `t` is computed using information only available after `t` (e.g. using the day's closing price to label a signal generated at 10am, or normalizing features using statistics computed over the whole dataset including the future). |
| 2 | Survivorship bias | Historical universe only includes tickers that still exist today, silently excluding delisted/bankrupt/acquired names — this systematically overstates historical returns. |
| 3 | Overfitting to a small/noisy sample | A model or rule set tuned on a few months or a narrow set of tickers picks up noise, not signal, and doesn't generalize. |
| 4 | Overfitting through repeated manual iteration | Even with a "correct" model-fitting procedure, running backtest → look at result → tweak a threshold or feature → re-run, dozens of times against the same historical stretch, fits the *strategy* to that stretch. No single model run is overfit, but the search process is. This is the most common way a backtest looks great and then loses money live. |
| 5 | Ignoring transaction costs | Commission, per-share fees, and bid-ask spread eaten on every entry and exit; a strategy with a small edge per trade can have that edge entirely consumed by costs. |
| 6 | Unrealistic fill assumptions | Assuming every signal fills instantly at the exact signal price, ignoring slippage, ignoring that limit orders can go unfilled, ignoring latency between signal and order submission. |
| 7 | Optimizing for win rate instead of expectancy | Win rate is intuitive but not sufficient. `expectancy = win_rate × avg_win − loss_rate × avg_loss`. A 70%-win-rate system with rare huge losses can be a net loser; a 33%-win-rate system with disciplined small losses and big winners can be strongly profitable. Chasing win rate directly is a common way retail systems get *worse*. |
| 8 | No walk-forward validation | Random k-fold cross-validation on time-series data leaks future information into training (a model trained partly on "future" rows relative to a test row will look better than it is). Backtest and live performance then diverge sharply. |
| 9 | No hard risk limits | No daily loss limit, no max-drawdown circuit breaker, no per-trade stop — a losing streak or a single bad trade can do outsized damage with nothing to stop it. |

---

## 2. How this project's architecture guards against each one

This is the spec. No trading code is written yet — each phase below is scoped to build the corresponding guard.

1. **Lookahead bias** — `src/features/` (Phase 3) computes every feature using only data at or before `t`. An automated test shifts the dataset forward and asserts that features at time `t` don't change when future rows are altered. Any dataset-wide normalization (e.g. z-scores) is fit only on training folds, never on validation/test or the full dataset.
2. **Survivorship bias** — `src/data/` (Phase 2) documents explicitly where the ticker universe comes from and whether delisted names are included. If the data source is survivorship-biased (common with free-tier data), this is recorded as a known limitation in the README rather than silently ignored, so it's visible when interpreting backtest results.
3. **Overfitting to a small sample** — Phase 4 requires walk-forward (rolling-origin) validation across multiple historical regimes (§3 of this doc, and explicit regime testing in Phase 4), plus a naive baseline comparison (buy-and-hold, random entries with the same risk parameters) so we can tell if the model adds value beyond chance.
4. **Overfitting through iteration** — An append-only experiment log (see §5) records every backtest run: date, git commit, parameters, data range, and result metrics. A **final holdout period is set aside now (§5) and not touched until Phase 8** — no matter how many times earlier periods get re-tested during development. This makes the iteration-overfitting failure mode visible after the fact even if it happens during development.
5. **Transaction costs** — Phase 5's backtester models Alpaca's actual commission/fee schedule and bid-ask spread, not just the raw signal price.
6. **Unrealistic fills** — Phase 5 fills on the bar *after* the signal (not the signal bar's own close), applies a configurable latency delay, models market vs. limit orders including partial/no fills, and uses a conservative (not favorable-case) assumption for resolving stop-loss vs. take-profit ambiguity within a single OHLC bar.
7. **Win rate vs. expectancy** — See §3: the target metric is expectancy per trade and risk-adjusted return, not win rate. Every metrics report (Phases 4, 5, 8, 9) surfaces expectancy, Sharpe, Sortino, and max drawdown alongside win rate so win rate is never viewed in isolation.
8. **No walk-forward validation** — Phase 4 mandates rolling-origin validation with purging/embargo between train and validation windows (so overlapping triple-barrier labels near a split boundary don't leak future information into training).
9. **No hard risk limits** — Phase 6 builds a standalone risk module used by *both* backtest and live execution (not duplicated/reimplemented for live), with per-trade stops, a daily loss limit, a max-drawdown circuit breaker, and PDT-aware trade-count tracking (§4).

---

## 3. Target metric

**Primary: expectancy per trade.** `expectancy = win_rate × avg_win − loss_rate × avg_loss`. This is the number that actually determines whether the system makes money, and it's robust to the win-rate trap described above (a low win rate with a strongly positive expectancy is a good system; a high win rate with negative expectancy is not).

**Secondary: Sharpe ratio of the equity curve** (and Sortino, which only penalizes downside volatility — often more relevant for a strategy with defined stop-losses). This measures *risk-adjusted* return: a system with high expectancy but wild equity-curve swings is harder to run in practice (harder to stay disciplined through drawdowns, harder to size sensibly) than one with a smoother curve, even at similar total return.

**Tertiary / diagnostic only: win rate, profit factor, max drawdown, average trade duration.** These are reported in every metrics output (Phases 4, 5, 8, 9) for diagnosis, but never used as the optimization target. Max drawdown specifically feeds Phase 6's circuit breaker design.

Win rate alone is explicitly *not* a target metric anywhere in this system. If a change to the model or strategy improves win rate but degrades expectancy or Sharpe, that change is rejected.

---

## 4. Pattern Day Trader (PDT) rule and strategy shape

**Confirmed: starting account equity is under $25,000.**

This means the account is subject to the PDT rule on margin: **no more than 3 day trades in any rolling 5-business-day window**, or the account risks a trading restriction/freeze. This applies identically on Alpaca and Robinhood — it's a FINRA/exchange rule on margin accounts, not a broker-specific one.

**Decision (confirmed with user): strategy shape is a hybrid, chosen because account size is under $25k.** Pure intraday day-trading, unconstrained, would hit the 3-trade ceiling almost immediately for any reasonably active signal generator — at that point the model's signal quality becomes irrelevant because the system is capacity-constrained by the trade counter, not by edge. Concretely, this means:

- The triple-barrier labeling in Phase 3 and the position-holding logic in Phase 5/6 must support a **configurable max holding time longer than one trading day**, not just intraday exits.
- The risk module (Phase 6) tracks day-trade count in a rolling 5-business-day window and **blocks a new same-day round trip once the count would hit 3**, rather than blocking new positions outright — the system can still open a position and carry it overnight/multi-day when the day-trade budget is exhausted.
- The signal/strategy design should not assume every entry closes same-day; features and models should be equally valid for a position held 1–3 days as for one held under an hour, since the risk layer — not the strategy — decides at execution time whether a given exit must be deferred past the same session to avoid a PDT violation.
- This threshold ($25k) is a configurable constant in the risk module (Phase 6), not hardcoded, so the constraint can be relaxed automatically if/when equity grows past it.

This is a design input starting now, not a constraint discovered after Phase 6 is built.

---

## 5. Experiment logging and the final holdout

**Experiment log:** `experiment_log.csv` (repo root, append-only, git-tracked) with one row per backtest/training run:

```
run_id, date, git_commit, phase, data_start, data_end, params_json, win_rate, expectancy, sharpe, sortino, max_drawdown, notes
```

Every run in Phase 4 (training) and Phase 5 (backtesting) appends a row before moving on — including runs whose results were "bad" and simply discarded from consideration. The point of the log is to make the iteration process itself auditable: if the final strategy's backtest numbers look too good, the log lets us check how many variations were tried to get there.

**Final holdout:** A contiguous historical period, chosen *now* and recorded below, is set aside and excluded from all data used in Phases 2–7 (data exploration, feature development, training, walk-forward validation, and iterative backtesting). It is used exactly once, in Phase 8, as the last check before considering real-money deployment. If Phase 8 holdout performance is materially worse than the walk-forward validation performance from Phase 4, that is a signal of overfitting through iteration, and the correct response is to revisit the process (and account for it in the go/no-go decision) — not to retune against the holdout and re-check it.

- **Holdout period (pinned 2026-08-10, Phase 2):** `2026-05-01` through `2026-07-31` inclusive — the most recent complete 3 calendar months as of when Phase 2's live Alpaca connection was validated. No feature development, training, walk-forward validation, or backtest iteration in Phases 3–7 may read data from this window. It is used exactly once, in Phase 8.
- **Rule:** no code in Phases 2–7 reads, prints, plots, or otherwise inspects data inside the holdout window. This should be enforced mechanically where practical (e.g. the data-loading utility takes an explicit `exclude_holdout: bool` and defaults to `True`).

---

## Open items for user confirmation

- [x] Starting account equity: **under $25,000** → PDT rule applies.
- [x] Strategy shape: **hybrid** (day trades + occasional multi-day swing holds), driven by the PDT constraint above.
- [x] Exact holdout date range: **2026-05-01 through 2026-07-31**, pinned in Phase 2.
