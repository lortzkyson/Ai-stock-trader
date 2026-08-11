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
