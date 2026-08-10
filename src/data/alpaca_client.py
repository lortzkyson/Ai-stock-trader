"""Thin wrapper around Alpaca's historical market-data API.

Feed decision (see docs/data_feed_decision.md): historical bar queries
default to `feed=DataFeed.SIP`, the full consolidated tape (~100% of US
equity volume). Alpaca's free plan allows historical SIP queries as long as
the query's `end` timestamp is at least 15 minutes in the past — always true
for backtesting, so no paid subscription is required for this module.
Real-time/live data for paper or live trading (Phase 7+) is a separate
decision: the free plan's live stream is IEX-only (~2.5% of volume). This
module does not make that call.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, time

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.models.bars import BarSet
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


def _to_datetime(value: date | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.combine(value, time.min)


@dataclass(frozen=True)
class BarsQuery:
    symbols: list[str]
    start: date | datetime
    end: date | datetime
    timeframe: TimeFrame = TimeFrame(1, TimeFrameUnit.Minute)
    feed: DataFeed = DataFeed.SIP
    adjustment: Adjustment = Adjustment.ALL


class AlpacaBarsClient:
    """Wraps alpaca-py's StockHistoricalDataClient so callers don't import it directly."""

    def __init__(self, api_key: str | None = None, secret_key: str | None = None) -> None:
        api_key = api_key or os.environ["ALPACA_API_KEY"]
        secret_key = secret_key or os.environ["ALPACA_SECRET_KEY"]
        self._client = StockHistoricalDataClient(api_key, secret_key)

    def fetch_bars(self, query: BarsQuery) -> pd.DataFrame:
        request = StockBarsRequest(
            symbol_or_symbols=query.symbols,
            timeframe=query.timeframe,
            start=_to_datetime(query.start),
            end=_to_datetime(query.end),
            feed=query.feed,
            adjustment=query.adjustment,
        )
        bars = self._client.get_stock_bars(request)
        if not isinstance(bars, BarSet):
            raise TypeError(f"expected BarSet from get_stock_bars, got {type(bars)}")
        return bars.df.reset_index()
