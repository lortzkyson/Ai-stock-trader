#!/usr/bin/env python3
"""Sweep cash-secured put moneyness from deep-OTM to ATM.

This is a parameter sweep, which is the overfitting trap docs/pre-mortem.md §5
exists to prevent. It is defensible here only because the goal is to map the
whole risk/return curve, not to find the level with the best backtest number.
The full curve is always reported. Picking the winning row and trading it would
be exactly the mistake.

Every configuration is also run through an explicit crash scenario, because the
available option history (2024-01-18 onward) contains no 2008/2020-style event —
and that tail is where short-volatility strategies actually lose. A backtest
that cannot show the defining risk of the strategy has to be supplemented with
one that can.
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import Adjustment, DataFeed
from dotenv import load_dotenv
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

MONEYNESS_LEVELS = [1.00, 0.99, 0.98, 0.97, 0.95, 0.92, 0.90]
CRASH_SCENARIOS = [-0.10, -0.20, -0.30]
START, END = date(2024, 2, 1), date(2026, 5, 1)


def occ(underlying: str, exp: date, opt_type: str, strike: float) -> str:
    return f"{underlying}{exp:%y%m%d}{opt_type}{int(round(strike * 1000)):08d}"


def third_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    while d.weekday() != 4:
        d += timedelta(days=1)
    return d + timedelta(days=14)


def main() -> int:
    load_dotenv()
    key, secret = os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"]
    sc = StockHistoricalDataClient(key, secret)
    oc = OptionHistoricalDataClient(key, secret)

    spy = sc.get_stock_bars(
        StockBarsRequest(
            symbol_or_symbols=["SPY"], timeframe=TimeFrame(1, TimeFrameUnit.Day),
            start=datetime(2023, 12, 1, tzinfo=timezone.utc),
            end=datetime(2026, 8, 10, tzinfo=timezone.utc),
            feed=DataFeed.SIP, adjustment=Adjustment.ALL,
        )
    ).df.reset_index()
    spy["d"] = pd.to_datetime(spy["timestamp"]).dt.date
    px = dict(zip(spy["d"], spy["close"]))

    expiries = []
    for y in range(2024, 2027):
        for m in range(1, 13):
            e = third_friday(y, m)
            if START <= e <= END:
                expiries.append(e)

    summary = []
    for moneyness in MONEYNESS_LEVELS:
        rows = []
        for exp in expiries:
            entry = exp - timedelta(days=30)
            while entry not in px and entry < exp:
                entry += timedelta(days=1)
            if entry not in px:
                continue
            spot = px[entry]
            strike = round(spot * moneyness)
            sym = occ("SPY", exp, "P", strike)
            try:
                bars = oc.get_option_bars(
                    OptionBarsRequest(
                        symbol_or_symbols=[sym], timeframe=TimeFrame(1, TimeFrameUnit.Day),
                        start=datetime(entry.year, entry.month, entry.day, tzinfo=timezone.utc),
                        end=datetime(exp.year, exp.month, exp.day, tzinfo=timezone.utc),
                    )
                ).df
                if not len(bars):
                    continue
                b = bars.reset_index()
                b["d"] = pd.to_datetime(b["timestamp"]).dt.date
                first = b[b["d"] == entry]
                if not len(first):
                    continue
                premium = float(first["close"].iloc[0])
            except Exception:
                continue

            settle = px.get(exp)
            if settle is None:
                continue
            pnl = premium - max(0.0, strike - settle)
            rows.append({
                "premium": premium, "strike": strike, "spot": spot,
                "assigned": settle < strike, "ret": pnl / strike,
                "premium_pct": premium / strike,
            })

        if len(rows) < 12:
            continue
        df = pd.DataFrame(rows)
        t_stat, p_value = stats.ttest_1samp(df["ret"], 0)
        mean_premium_pct = df["premium_pct"].mean()

        crash = {}
        for shock in CRASH_SCENARIOS:
            # Strike sits at `moneyness` x spot; a shock takes spot to (1+shock).
            intrinsic = max(0.0, moneyness - (1 + shock)) / moneyness
            crash[shock] = mean_premium_pct - intrinsic

        summary.append({
            "moneyness": moneyness, "n": len(df),
            "mean_prem_pct": mean_premium_pct * 100,
            "monthly_ret": df["ret"].mean() * 100,
            "annualized": df["ret"].mean() * 12 * 100,
            "win_rate": (df["ret"] > 0).mean() * 100,
            "assigned_pct": df["assigned"].mean() * 100,
            "worst": df["ret"].min() * 100,
            "p": p_value,
            **{f"crash{int(s*100)}": crash[s] * 100 for s in CRASH_SCENARIOS},
        })

    out = pd.DataFrame(summary)
    pd.set_option("display.width", 250)
    print("=== Cash-secured put moneyness sweep (SPY, 30DTE, held to expiry) ===\n")
    print(out[["moneyness", "n", "mean_prem_pct", "monthly_ret", "annualized",
               "win_rate", "assigned_pct", "worst", "p"]]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n=== Crash stress test: return in a single month if SPY drops X% ===")
    print("(the data window contains no such month; these are computed, not observed)\n")
    print(out[["moneyness", "annualized", "crash-10", "crash-20", "crash-30"]]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n=== Years of premium erased by one crash month ===\n")
    for _, r in out.iterrows():
        if r["annualized"] <= 0:
            continue
        yrs = {s: abs(r[f"crash{int(s*100)}"]) / r["annualized"] for s in CRASH_SCENARIOS}
        print(f"  {r['moneyness']:.2f} moneyness ({r['annualized']:5.2f}%/yr): "
              + "  ".join(f"{int(s*100)}% crash = {yrs[s]:4.1f} yrs" for s in CRASH_SCENARIOS))

    first_spot = px[min(d for d in px if d >= START - timedelta(days=35))]
    last_spot = px[max(d for d in px if d <= END)]
    print(f"\nSPY buy-and-hold over the same window: {(last_spot/first_spot-1)*100:+.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
