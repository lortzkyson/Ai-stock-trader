#!/usr/bin/env python3
"""Backtest cash-secured puts (5% OTM, 30DTE, held to expiry) on SPY.

Result: statistically significant (p=0.037, 96% win rate) but +3.66%/yr against
SPY buy-and-hold at +46% over the same window. The premium is real and the
strategy reliably harvests it; the premium is simply small relative to the
capital cash-securing ties up, and is compensation for a tail risk this data
window does not contain.
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
from scipy import stats

K=os.environ['ALPACA_API_KEY']; S=os.environ['ALPACA_SECRET_KEY']
sc=StockHistoricalDataClient(K,S); oc=OptionHistoricalDataClient(K,S)
spy=sc.get_stock_bars(StockBarsRequest(symbol_or_symbols=['SPY'],timeframe=TimeFrame(1,TimeFrameUnit.Day),
    start=datetime(2023,12,1,tzinfo=timezone.utc),end=datetime(2026,8,10,tzinfo=timezone.utc),
    feed=DataFeed.SIP,adjustment=Adjustment.ALL)).df.reset_index()
spy['d']=pd.to_datetime(spy['timestamp']).dt.date
px=dict(zip(spy['d'],spy['close']))

def occ(u,exp,t,k): return f'{u}{exp:%y%m%d}{t}{int(round(k*1000)):08d}'
def third_friday(y,m):
    d=date(y,m,1)
    while d.weekday()!=4: d+=timedelta(days=1)
    return d+timedelta(days=14)

# Cash-secured put, ~5% out of the money, 30 DTE, held to expiry.
OTM=0.95
rows=[]
for y in range(2024,2027):
    for m in range(1,13):
        exp=third_friday(y,m)
        if not (date(2024,2,1)<=exp<=date(2026,5,1)): continue
        entry=exp-timedelta(days=30)
        while entry not in px and entry<exp: entry+=timedelta(days=1)
        if entry not in px: continue
        spot=px[entry]; strike=round(spot*OTM)
        sym=occ('SPY',exp,'P',strike)
        try:
            b=oc.get_option_bars(OptionBarsRequest(symbol_or_symbols=[sym],
                timeframe=TimeFrame(1,TimeFrameUnit.Day),
                start=datetime(entry.year,entry.month,entry.day,tzinfo=timezone.utc),
                end=datetime(exp.year,exp.month,exp.day,tzinfo=timezone.utc))).df
            if not len(b): continue
            bb=b.reset_index(); bb['d']=pd.to_datetime(bb['timestamp']).dt.date
            r0=bb[bb['d']==entry]
            if not len(r0): continue
            prem=float(r0['close'].iloc[0])
        except Exception: continue
        settle=px.get(exp)
        if settle is None: continue
        assigned_loss=max(0.0, strike-settle)
        pnl=prem-assigned_loss
        capital=strike*100
        rows.append({'expiry':exp,'spot':spot,'strike':strike,'premium':prem,
                     'settle':settle,'assigned':settle<strike,'pnl_per_share':pnl,
                     'pnl_contract':pnl*100,'capital':capital,'ret_on_capital':pnl/strike})
df=pd.DataFrame(rows)
print(f"cash-secured puts (5% OTM, 30DTE, held to expiry): {len(df)} months\n")
pd.set_option('display.width',200)
print(df[['expiry','spot','strike','premium','settle','assigned','pnl_per_share']].to_string(
    index=False,float_format=lambda x:f'{x:.2f}'))
n=len(df)
tot=df['ret_on_capital'].sum()
print(f"\nassigned (put finished ITM): {df['assigned'].sum()}/{n}")
print(f"win rate: {(df['pnl_per_share']>0).mean()*100:.0f}%")
print(f"mean monthly return on capital: {df['ret_on_capital'].mean()*100:+.3f}%")
print(f"cumulative return on capital over {n} months: {tot*100:+.2f}%")
print(f"annualized (simple): {tot/ (n/12) *100:+.2f}%")
print(f"best month {df['ret_on_capital'].max()*100:+.2f}% | worst {df['ret_on_capital'].min()*100:+.2f}%")
t,p=stats.ttest_1samp(df['ret_on_capital'],0)
print(f"t={t:.2f} p={p:.4f}")
sp0=px[min(df['expiry'].min()-timedelta(days=30), min(px))] if False else None
first=df['spot'].iloc[0]; last=px[df['expiry'].iloc[-1]]
print(f"\nSPY buy-and-hold same window: {(last/first-1)*100:+.2f}%")
