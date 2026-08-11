# Next steps — picking this back up

## Current live state

The paper-trading loop is **running automatically** via a launchd agent (every 5 minutes, while the Mac is awake). It entered real paper positions on 2026-08-11. Nothing is at risk — paper money only, and no live-trading code exists in this repo.

To check on it:
```bash
make monitor
```
To stop it:
```bash
launchctl bootout gui/501/com.kysonlortz.ai-stock-trader.paper-loop
```

## Where the strategy work left off

Decision made: pursue a **documented anomaly with a structural reason to persist** (PEAD or cross-sectional momentum) instead of further pattern-searching on 1-minute bars. Rationale is in [go_live_review.md](go_live_review.md) §2 — three rounds of intraday feature engineering all failed to beat a random baseline, which is the expected outcome in the most heavily arbitraged corner of the market.

### What was verified before stopping

- **Daily bars: available and good.** Alpaca gives ~10.5 years (2016-01 → present) of SIP-quality adjusted daily bars, free. Far more statistical power than the 14 months of minute bars used so far.
- **Earnings data: NOT available from Alpaca.** Its corporate-actions endpoint covers splits, dividends, mergers, spinoffs, name changes — no earnings dates and no consensus estimates.

## RESULT: momentum ran. It does not show a demonstrated edge.

Full numbers in `reports/momentum_2026-08-11_213446.md`. Headline looks spectacular and is misleading:

| | total return | CAGR | vol | Sharpe |
|---|---|---|---|---|
| Momentum (20 names, monthly) | **+2,140%** | 35.2% | 40.4% | 0.95 |
| SPY buy-and-hold | +315% | 14.8% | 17.6% | 0.87 |
| **SPY levered 2.29x to match vol** | **+1,500%** | 30.9% | 40.4% | 0.87 |

**Finding 1 — the outperformance is risk, not skill.** The strategy runs at 2.29x SPY's volatility. Simply leveraging SPY to the same volatility captures +1,500% of the +2,140%. Excess over the *risk-matched* benchmark: **t=0.28, p=0.78** — no detectable edge at all. Against unlevered SPY it looked significant (p=0.026); that comparison was just measuring the decision to take more risk. Sharpe 0.95 vs 0.87 was the tell all along.

**Finding 2 — survivorship bias is likely fatal here, not marginal.** The most-held names are exactly the cohort that delists: VKTX/MDGL/ARWR/AXSM (speculative biotech), RIOT/MARA (crypto miners), BTU/AMR (coal), CVNA. The universe contains only companies still listed in 2026, so the ones from this cohort that went to zero are invisible. Sensitivity to an assumed annual delisting rate among holdings:

| annual delist rate | CAGR | p vs SPY |
|---|---|---|
| 0% (as backtested) | 35.4% | 0.025 ✓ |
| 2% | 32.8% | 0.042 ✓ |
| **5%** | 28.8% | **0.084 ✗** |
| 10% | 22.5% | 0.226 ✗ |
| 15% | 16.5% | 0.494 ✗ |

Significance evaporates by 5%/yr. For speculative biotech and crypto miners over 2016-2026, the real rate was very likely higher than that.

**What did survive scrutiny** (worth noting, not dismissing): it beat SPY in 9 of 10 years including down years (2018 +5.0% vs -5.0%; 2022 -4.6% vs -18.2%), held 432 distinct names so it isn't one lucky pick, and the monthly-frequency t-test matched the daily one (p=0.025 vs 0.026), so the significance wasn't an artifact of overlapping observations. The Sharpe edge is positive — just small and not statistically distinguishable from zero.

**Verdict: NO-GO on this as a live strategy.** A 2.29x-beta portfolio of speculative small caps, measured in a survivorship-biased universe, whose risk-adjusted excess is indistinguishable from zero.

**If continuing down this path**, the highest-value next step is fixing survivorship bias — sourcing a point-in-time constituent list (delisted price history *is* available from Alpaca; only the membership list is missing). Without that, no equity backtest here can be trusted, and that limitation now has a measured magnitude rather than a hand-wave.

---

### RESOLVED — ETF contamination (fixed 2026-08-11)

**The screened universe contains ETFs, including leveraged single-stock ETFs.** Filtering to NYSE+NASDAQ (to exclude ARCA, which is predominantly ETFs) was not sufficient — plenty of ETFs list on NASDAQ/NYSE. Confirmed in the cached universe: `AAXJ` (iShares Asia ex-Japan) and `AAPU` (a 2x leveraged AAPL ETF), among 2,204 symbols.

This is not cosmetic. Leveraged ETFs mechanically exhibit extreme trailing returns, so a top-20 momentum ranking would likely fill up with them — turning a "buy strong stocks" strategy into "buy whatever was most leveraged into the last rally," which has completely different risk characteristics and well-documented volatility decay. **Any backtest run before this is fixed is measuring the wrong strategy.**

Fix options, in order of preference:
1. Check whether Alpaca's `Asset` model exposes anything usable to distinguish ETFs from common stock (worth inspecting `Asset` fields and `attributes` — not yet checked).
2. Failing that, filter by name pattern (ETF issuers: iShares, ProShares, Direxion, SPDR, Invesco, Vanguard, etc.) — crude and leaky, but better than nothing.
3. Source a proper security-type list from another provider.

### Progress so far (committed, not yet run end-to-end)

Three new modules are built, linted, typechecked, and committed — but **the strategy has not been backtested yet**. Nothing here has produced a number, so nothing here is evidence of anything:

- `src/data/daily_bars.py` — batched daily-bar fetching + panel caching
- `src/strategies/momentum.py` — canonical 12-1 cross-sectional momentum
- `src/backtest/portfolio.py` — monthly-rebalance portfolio backtester

**Status of the run:**
1. ✅ Universe screening — **done and cached** to `data/cache/momentum_universe.txt` (2,204 symbols, union of 4 point-in-time screens). Took ~11 minutes. But see the ETF contamination issue above.
2. ❌ Fetch ~10 years of daily bars — not started. The run was stopped here.
3. ✅ Runner script (`scripts/run_momentum_backtest.py`).
4. ✅ Benchmarks + paired t-test vs SPY and equal-weight buy-and-hold.
5. ✅ Tests — 25 added, 158 passing overall.

**To resume:** fix the ETF issue, delete `data/cache/momentum_universe.txt` to force a re-screen, then rerun. Add `PYTHONUNBUFFERED=1` so progress is visible — the first run buffered all output and was impossible to monitor.

Expect ~30-50 minutes total; it's network-bound, not CPU-bound.

### The fork to resolve first

**PEAD needs an earnings surprise measure**, which normally requires consensus EPS estimates (paid data). Two workarounds:
1. Use the **announcement-window abnormal return** as the surprise proxy — well-founded in the literature, needs only earnings *dates* plus daily bars.
2. Source earnings dates from a free provider (e.g. `yfinance`). Note this is an unofficial Yahoo scraper with the same category of dependency risk we flagged for `robin_stocks` — it can break without warning.

**Cross-sectional momentum (12-1) needs no new data at all** — daily bars alone. Cheapest thing to test, and it validates the daily-horizon rebuild before taking on an external data dependency.

### Known design constraints for whichever path

- **Universe breadth matters.** Cross-sectional ranking over 30 symbols gives ~3 stocks per decile, which is statistically meaningless. Real momentum/PEAD studies use hundreds to thousands of names. Widening the universe will re-raise the survivorship-bias limitation already documented in the README.
- **Architecture changes needed.** The current triple-barrier / per-symbol / intraday design doesn't fit a cross-sectional strategy. Labeling, portfolio construction, and rebalance logic need rework. The data pipeline, risk engine, and execution layer largely carry over.
- **PDT stops mattering** at multi-week holds — a real side benefit.
- **The holdout is only 3 months** (2026-05-01 → 07-31). For a strategy holding positions 1-3 months, that's very few independent observations. It may need to be re-pinned to a longer window before it can meaningfully validate a daily-horizon strategy.

### The discipline that still applies

Three attempts have now been run against this data. Every additional tweak-and-retest cycle burns credibility (the iteration-overfitting trap from [pre-mortem.md](pre-mortem.md) §5). The untouched holdout can be spent **once**. Log every run in `experiment_log.csv` regardless of outcome.

Benchmark to beat: buy-and-hold returned **+27%** over the last test period; the intraday model returned **-4.8%**.
