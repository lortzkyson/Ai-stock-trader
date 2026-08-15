#!/usr/bin/env python3
"""Measure the volatility risk premium: ATM straddle cost vs realized move.

Data limit: Alpaca option history begins 2024-01-18 (verified - LEAPS listed in
2022 still return nothing before that date), so this covers ~24 monthly
observations in a period with no 2008/2020-style crash. That matters enormously
for short-volatility strategies, whose entire risk lives in exactly the tail
this window excludes.
"""

import os
import sys

sys.path.insert(0,'src')
from dotenv import load_dotenv; load_dotenv(dotenv_path=".env")
from datetime import date, datetime, timedelta, timezone

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

K=os.environ['ALPACA_API_KEY']; S=os.environ['ALPACA_SECRET_KEY']
sc=StockHistoricalDataClient(K,S); oc=OptionHistoricalDataClient(K,S)

spy=sc.get_stock_bars(StockBarsRequest(symbol_or_symbols=['SPY'],timeframe=TimeFrame(1,TimeFrameUnit.Day),
    start=datetime(2023,12,1,tzinfo=timezone.utc),end=datetime(2026,8,10,tzinfo=timezone.utc),
    feed=DataFeed.SIP,adjustment=Adjustment.ALL)).df.reset_index()
spy['d']=pd.to_datetime(spy['timestamp']).dt.date
px=dict(zip(spy['d'],spy['close']))
days=sorted(px)

def occ(u,exp,t,k): return f'{u}{exp:%y%m%d}{t}{int(round(k*1000)):08d}'
def third_friday(y,m):
    d=date(y,m,1)
    while d.weekday()!=4: d+=timedelta(days=1)
    return d+timedelta(days=14)

rows=[]
for y in range(2024,2027):
    for m in range(1,13):
        exp=third_friday(y,m)
        if not (date(2024,2,1) <= exp <= date(2026,5,1)): continue
        entry=exp-timedelta(days=30)
        while entry not in px and entry<exp: entry+=timedelta(days=1)
        if entry not in px: continue
        spot=px[entry]
        strike=round(spot)  # SPY strikes are $1 apart near the money
        legs={}
        for t in ('C','P'):
            sym=occ('SPY',exp,t,strike)
            try:
                b=oc.get_option_bars(OptionBarsRequest(symbol_or_symbols=[sym],
                    timeframe=TimeFrame(1,TimeFrameUnit.Day),
                    start=datetime(entry.year,entry.month,entry.day,tzinfo=timezone.utc),
                    end=datetime(exp.year,exp.month,exp.day,tzinfo=timezone.utc))).df
                if len(b):
                    bb=b.reset_index(); bb['d']=pd.to_datetime(bb['timestamp']).dt.date
                    r0=bb[bb['d']==entry]
                    if len(r0): legs[t]=float(r0['close'].iloc[0])
            except Exception: pass
        if len(legs)<2: continue
        straddle=legs['C']+legs['P']
        settle=px.get(exp)
        if settle is None: continue
        actual=abs(settle-spot)
        rows.append({'expiry':exp,'entry':entry,'spot':spot,'strike':strike,
                     'straddle_cost':straddle,'actual_move':actual,
                     'seller_pnl':straddle-actual,
                     'implied_pct':straddle/spot,'actual_pct':actual/spot})

df=pd.DataFrame(rows)
print(f"monthly ATM straddles measured: {len(df)}\n")
if len(df):
    pd.set_option('display.width',200)
    print(df[['expiry','spot','straddle_cost','actual_move','seller_pnl']].to_string(index=False,
          float_format=lambda x:f'{x:.2f}'))
    print(f"\nmean implied move: {df['implied_pct'].mean()*100:.2f}% of spot")
    print(f"mean actual move:  {df['actual_pct'].mean()*100:.2f}% of spot")
    print(f"volatility risk premium: {(df['implied_pct'].mean()-df['actual_pct'].mean())*100:+.2f} pp")
    print(f"\nstraddle SELLER: won {(df['seller_pnl']>0).sum()}/{len(df)} months "
          f"({(df['seller_pnl']>0).mean()*100:.0f}%)")
    print(f"  total P&L per 1 straddle: ${df['seller_pnl'].sum():+.2f}")
    print(f"  best month ${df['seller_pnl'].max():+.2f} | worst month ${df['seller_pnl'].min():+.2f}")
    from scipy import stats
    t,p=stats.ttest_1samp(df['seller_pnl'],0)
    print(f"  t={t:.2f} p={p:.3f}")
