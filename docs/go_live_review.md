# Go/No-Go Review — Phase 8

**Recommendation: NO-GO. Do not connect this system to a real-money account yet.**

This document is analysis only. No live-order-submission code exists anywhere in this repo, and none should be written until the blockers below are resolved and you've explicitly reviewed and approved moving forward — per the Phase 8 spec and independent of it, executing a real financial trade is something no agent should ever do on your behalf.

---

## 1. Paper trading track record

**There isn't one yet, and that's a hard blocker.** The spec calls for summarizing 4-6 weeks of paper trading results before considering go-live. What actually exists:

- Phase 7's execution code (`src/execution/`) was verified against the real Alpaca paper endpoint: a real limit order was submitted, confirmed in open orders, and cancelled; positions/equity were queried live. This proves the *plumbing* works.
- What did **not** happen: no sustained, unattended paper-trading run over days or weeks generating a real track record of live fills, slippage, and P&L to compare against the backtest.
- This is a deliberate scope decision, not an oversight — standing up a persistent trading loop that runs unattended for weeks wasn't something to start without your explicit go-ahead, especially stacked under a NO-GO on the model itself (§2). Building that multi-week track record is the next concrete step, and it's on you to kick off (e.g. via a scheduled task running `scripts/monitor.py` and whatever loop wraps signal generation → risk check → order submission), not something to retroactively fabricate here.

**Because there's no real paper track record, there's nothing to compare against the backtest for the "paper vs. backtest divergence" check this section is also supposed to do.** That check has to wait until real paper data exists.

One thing worth internalizing now, before that run starts: **Alpaca's paper fills tend to be more favorable than real ones** — less realistic slippage and rejection than a live account would see. Treat whatever paper results eventually come in as an optimistic upper bound, not a live-performance guarantee, and size any initial real-money ramp-up (§4) conservatively below what paper alone would suggest.

## 2. The model doesn't show a clear edge — this is the bigger blocker, and it's now been checked twice

**Round 1** (8 symbols, ~8 months, 8 base features): **[docs/model_card.md](model_card.md)**'s first pass found aggregate out-of-sample expectancy statistically indistinguishable from a random-entry baseline at the same trade frequency. **[reports/backtest_2026-08-10.md](../reports/backtest_2026-08-10.md)** then found the realistic backtest turned that already-thin signal net negative (expectancy -0.04%/trade, Sharpe -0.30) and flagged a real in-sample/out-of-sample divergence (overfitting signature).

**Round 2** (widened to 30 symbols, ~14 months, +3 features — minutes-since-open, RSI-14, range-position): the response to Round 1 wasn't to keep tweaking the same small run, it was one considered, larger attempt. It didn't help:
- Signal-level walk-forward (1.48M out-of-sample predictions): model expectancy (0.082%/trade) came in essentially tied with — marginally *below* — the random baseline (0.088%/trade), with a visible downward trend across folds (0.21% → 0.15% → 0.06% → 0.09% → 0.01%).
- The realistic backtest (**[reports/backtest_2026-08-11.md](../reports/backtest_2026-08-11.md)**) initially looked *better* this time — positive expectancy, Sharpe 1.12, +7.7% return — which was a genuine reversal worth taking seriously rather than accepting at face value. A one-sample t-test on the 310 real trades came back **p=0.33** (not remotely significant), and the profit was concentrated: one symbol accounted for 29% of all gross positive P&L, and fewer than half the traded symbols (14 of 29) were net positive. That's the signature of a small, noisy sample producing a lucky-looking headline number, not a demonstrated edge — and it's consistent with the much larger 1.48M-prediction sample showing no edge at all.

Two independent, differently-scoped attempts (different universe size, different history length, different feature set) both concluded the same thing: **this signal, from basic price/volume-derived features at 1-minute resolution on large-cap equities, isn't demonstrated to beat chance.** That's a more informative and more trustworthy result than either attempt alone — a real edge should have shown up more clearly with more data and more features, not stayed flat or gotten murkier.

None of this is a reason to distrust the pipeline — it's doing exactly what Phase 0's pre-mortem designed it to do, including catching a misleadingly good-looking headline number (Round 2's backtest) before it could be mistaken for validation. Going live now would mean putting real capital behind a signal that has been checked twice and hasn't beaten chance either time.

## 3. Circuit breakers

The spec asks to confirm every circuit breaker fired at least once in paper testing, simulating if it hasn't happened naturally. Since no live/paper trading run has happened yet (§1), none have fired under real conditions. What *has* happened:

- **Unit-tested and confirmed correct** (`tests/risk/test_engine.py`): a simulated sustained losing streak trips the max-drawdown breaker and blocks all further entries until an explicit `manual_reset()`; a simulated bad day trips the daily loss limit and resets automatically the next day; a simulated 4-trade sequence under $25k equity correctly blocks the 4th same-day round trip via the PDT tracker.
- **Not yet confirmed**: that these fire correctly against a live/paper order flow with real (if delayed/optimistic) fills, real equity updates arriving asynchronously, real API latency, etc. — only the logic itself has been exercised, not the full live integration path.

This should be explicitly re-verified once the paper-trading run in §1 exists — e.g. by deliberately engineering a losing stretch (small size, short window) early in that run and confirming the breaker actually halts new orders in the live loop, not just in a unit test.

## 4. Alpaca vs. Robinhood

Alpaca supports real-money live trading through the same API already used for paper trading and historical data in this repo — going live would mean pointing `ALPACA_BASE_URL` at `https://api.alpaca.markets` and using a live key/secret, with no changes to `src/data/`, `src/features/`, `src/models/`, `src/backtest/`, or `src/risk/`. `src/execution/client.py` currently hard-refuses `paper=False` by design (see its docstring) — that guard would need to be deliberately removed as part of whatever go-live process you choose, and should probably be replaced with something stronger than its own absence (an explicit env var confirmation, a separate reviewed PR, etc.) rather than just deleted.

Robinhood has no official trading API. The only way to automate it is an unofficial, reverse-engineered library (commonly `robin_stocks`), which:
- can break without warning whenever Robinhood changes its (undocumented, unsupported) internal API,
- carries real risk of account flagging or restriction for automated access,
- would require building and maintaining a second execution layer parallel to `src/execution/client.py`, duplicating everything Phase 6/7 already built for Alpaca.

**Recommendation: if/when you go live, do it on Alpaca directly.** There's no technical reason to introduce Robinhood into this system at all — keep Robinhood as your separate, manual, personal account, entirely outside this codebase, exactly as the original project brief suggested. This is your call, not mine to make for you, but I'm not aware of an argument for the unofficial-library path that outweighs its risks here.

## 5. If and when you do go live: a conservative ramp-up plan

This is offered for when the blockers above are actually resolved (a demonstrated edge in a widened/improved model, and a real multi-week paper track record that roughly matches the backtest) — not a plan to execute now.

1. **Start at the smallest size Alpaca allows** (fractional shares if needed, or a handful of shares of a single liquid symbol), with the same risk config already in `src/risk/engine.py` (1% risked per trade, 25% max position, 2% daily loss limit, 10% max drawdown).
2. **Run live at that size for a fixed evaluation window** (e.g. 2-4 weeks) before any size increase, explicitly comparing live expectancy/Sharpe/drawdown against both the backtest and the paper run.
3. **Increase size only on a defined trigger** (e.g. live expectancy within some tolerance of paper expectancy for N consecutive weeks), never on a discretionary "it's been going well" basis — this is exactly the kind of manual iteration-without-a-rule the pre-mortem's experiment-log discipline exists to guard against.
4. **Keep the kill switch (`scripts/kill_switch.py`) and the automatic circuit breakers as the two independent stops** they were designed to be — the ramp-up plan doesn't replace either.
5. Revisit §2 first. Sizing up a plan for an edge that hasn't been shown yet just risks capital faster, not more safely.

---

## Summary

| Gate | Status |
|---|---|
| Multi-week paper track record | **Not done** — no persistent run has happened |
| Paper-vs-backtest divergence check | **Blocked** on the above |
| Model shows a demonstrated edge | **No, checked twice** — indistinguishable from random baseline in both an 8-symbol and a 30-symbol run; a good-looking Round 2 backtest number turned out to be statistically insignificant noise (p=0.33) |
| Circuit breakers verified | Unit-tested and correct; **not yet verified live** |
| Alpaca vs. Robinhood | Alpaca recommended if/when going live; decision is yours |
| Ramp-up plan | Drafted, contingent on the above |

**Bottom line:** the honest result of Phases 0-7 is that this particular model, on this particular universe and date range, hasn't earned the right to trade real money yet. That's a legitimate, useful outcome of doing this rigorously rather than a failure of the process — the alternative (skipping straight to live trading on an unvalidated edge) is exactly the failure mode this whole project was set up to avoid.
