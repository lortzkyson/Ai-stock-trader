from __future__ import annotations

from datetime import date

from alpaca.data.enums import Adjustment, DataFeed

from data.alpaca_client import AlpacaBarsClient, BarsQuery


def test_bars_query_defaults_to_full_sip_and_split_dividend_adjustment() -> None:
    query = BarsQuery(symbols=["AAPL"], start=date(2026, 1, 1), end=date(2026, 1, 2))
    assert query.feed == DataFeed.SIP
    assert query.adjustment == Adjustment.ALL


def test_client_construction_does_not_hit_network() -> None:
    # Dummy credentials: StockHistoricalDataClient only stores them locally,
    # it doesn't validate against Alpaca until an actual request is made.
    client = AlpacaBarsClient(api_key="dummy", secret_key="dummy")
    assert client is not None
