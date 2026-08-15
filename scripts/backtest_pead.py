#!/usr/bin/env python3
"""Post-earnings announcement drift test. See src/strategies/pead.py.

Result: no PEAD. 10,572 events, top-minus-bottom SUE bucket -0.53pp, p=0.44.

Two measurement problems were found and fixed before this number was trusted:
percentage surprise sorts on small EPS denominators rather than surprise size,
and hand-picking mega-caps made every bucket positive. Even corrected, the
sample carries survivorship bias - the earnings source only covers live
companies (162 of 400 sampled symbols had no data, i.e. the delisted ones),
which is why overall abnormal return is +1.54pp rather than ~0. That inflates
all buckets equally, so top-minus-bottom remains the trustworthy statistic.
"""

import os, sys, warnings, time, random
sys.path.insert(0,'src'); warnings.filterwarnings('ignore')
from dotenv import load_dotenv; load_dotenv(dotenv_path=".env")
from datetime import date
import pandas as pd, numpy as np, yfinance as yf
from scipy import stats
from alpaca.data.historical import StockHistoricalDataClient
from data.daily_bars import load_or_fetch_daily_panel, clean_daily_panel
from strategies.pead import (PeadConfig, align_to_trading_day, compute_event_drift,
                             compute_sue, bucket_by_surprise, summarize_by_bucket)

# Random sample from the survivorship-corrected universe rather than hand-picked
# mega-caps: the first pass selected today's winners and every bucket showed
# positive abnormal return, which was survivorship, not drift.
uni=[s for s in open('data/cache/momentum_universe_survivorship_corrected.txt').read().split() if s]
random.seed(42)
SYMS=sorted(random.sample(uni, 400))
print(f'sampled {len(SYMS)} symbols from {len(uni)}-name corrected universe')

c=StockHistoricalDataClient(os.environ['ALPACA_API_KEY'],os.environ['ALPACA_SECRET_KEY'])
panel=clean_daily_panel(load_or_fetch_daily_panel(c, sorted(set(SYMS)|{'SPY'}), date(2016,1,1), date(2026,4,30)))
px=panel.pivot_table(index='timestamp',columns='symbol',values='close',aggfunc='last').sort_index()
px.index=pd.to_datetime(px.index).tz_localize(None)
spy=px['SPY']; tdays=px.index
print(f'price panel: {len(px.columns)} symbols')

cfg=PeadConfig(); rows=[]; t0=time.time(); miss=0
for n,s in enumerate(SYMS):
    if s not in px.columns: continue
    try: ed=yf.Ticker(s).get_earnings_dates(limit=100)
    except Exception: miss+=1; continue
    if ed is None or not len(ed): miss+=1; continue
    ser=px[s].dropna().reindex(tdays)
    for ts,r in ed.iterrows():
        est,rep = r.get('EPS Estimate'), r.get('Reported EPS')
        if pd.isna(est) or pd.isna(rep): continue
        ann=pd.Timestamp(ts).tz_localize(None)
        rd=align_to_trading_day(ann,tdays)
        if rd is None: continue
        d=compute_event_drift(ser,spy,rd,cfg)
        if d is None: continue
        d.update({'symbol':s,'announced':ann,'estimate_eps':float(est),'reported_eps':float(rep)})
        rows.append(d)
    if (n+1)%100==0: print(f'  {n+1}/{len(SYMS)} symbols, {len(rows)} events, {time.time()-t0:.0f}s')

ev=pd.DataFrame(rows)
print(f'\nevents: {len(ev)} from {ev["symbol"].nunique()} symbols ({miss} symbols had no data)')
ev=compute_sue(ev)
b=bucket_by_surprise(ev,cfg)
print(f'events with usable SUE: {len(b)}\n')
pd.set_option('display.width',220)
print(summarize_by_bucket(b).to_string(index=False,float_format=lambda x:f'{x:.4f}'))
lo=b[b['bucket']==1]['abnormal_return']; hi=b[b['bucket']==cfg.n_buckets]['abnormal_return']
t,p=stats.ttest_ind(hi,lo,equal_var=False)
print(f"\nTOP-minus-BOTTOM SUE bucket: {(hi.mean()-lo.mean())*100:+.2f}pp   t={t:.2f}  p={p:.4f}")
print(f"overall mean abnormal return (all events): {b['abnormal_return'].mean()*100:+.2f}pp")
print("  ^ should be ~0 if the universe is unbiased; strongly positive implies survivorship")
