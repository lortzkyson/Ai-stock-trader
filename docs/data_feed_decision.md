# Data feed decision: Alpaca IEX vs SIP

**Decision: use Alpaca alone, with `feed=SIP` for all historical/backtest queries. No paid tier or alternative provider (e.g. Polygon.io) needed for Phase 2–5.**

## Background

Alpaca's free plan only includes *real-time* data from IEX, a single exchange covering roughly 2.5% of US equity volume — the consolidated tape (SIP, ~100% of volume) requires the paid AlgoTrader Plus plan for real-time access. If that limitation carried over to historical data, minute-bar backtests built on it would be working from a thin, potentially unrepresentative slice of the tape.

## Finding

It doesn't carry over. Historical queries are a separate case: Alpaca's free plan can query `feed=sip` historical bars as long as the query's `end` timestamp is **at least 15 minutes old**. Since backtesting by definition only ever queries the past, this restriction never actually binds for Phase 2 (data pipeline) or Phase 5 (backtesting) — `src/data/alpaca_client.py` defaults every `BarsQuery` to `DataFeed.SIP`, `Adjustment.ALL` (splits + dividends applied) at no extra cost.

## What this means for later phases

- **Phases 2–5 (data, features, models, backtest):** full consolidated-tape historical data, free. No action needed.
- **Phase 7+ (paper/live execution):** this finding does *not* apply. Real-time signal generation on the free plan is still IEX-only. That's a separate decision to make in Phase 7, when the tradeoff is: stay on IEX-only real-time data (cheap, ~2.5% of volume, may diverge from what the backtester assumed), or upgrade to AlgoTrader Plus for real-time SIP (matches the backtest's data assumptions exactly, at cost). Flagging it here so it isn't a surprise later — not deciding it now.

## Sources

- [Market Data FAQ](https://docs.alpaca.markets/us/docs/market-data-faq) — Alpaca docs, confirms the free-plan real-time IEX-only restriction and the 15-minutes-old exception for historical SIP queries.
- [About Market Data API](https://docs.alpaca.markets/us/docs/about-market-data-api)
