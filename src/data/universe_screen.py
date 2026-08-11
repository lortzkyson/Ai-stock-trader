"""Build a tradable candidate universe for cross-sectional strategies.

Two distinct filters are at work, and conflating them would be a mistake:

1. **This module** picks the *candidate* set — which symbols are worth fetching
   ten years of history for at all. It exists for practical reasons (fetching
   all ~8,500 NYSE/NASDAQ names over a decade is slow and mostly junk).
2. **`strategies/momentum.select_positions`** applies the real, point-in-time
   liquidity filter at each rebalance date using only trailing data. That's
   what actually governs investability during the backtest.

Selection bias: screening on a single recent window would pick names partly
*because* they were liquid at the end of the backtest period — information not
available at the start. To limit that, screening runs at several points across
the period and takes the union, approximating a point-in-time universe. This
does not fix survivorship bias (delisted names are absent from Alpaca's asset
list entirely — see docs/next_steps.md), only the "liquid today" selection
effect.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, cast

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.trading.requests import GetAssetsRequest

from data.daily_bars import fetch_daily_bars

MAJOR_STOCK_EXCHANGES = ("AssetExchange.NYSE", "AssetExchange.NASDAQ")


@dataclass(frozen=True)
class ScreenConfig:
    min_price: float = 5.0
    min_median_dollar_volume: float = 5_000_000.0
    window_days: int = 90
    max_symbols_per_screen: int = 1500


def list_candidate_symbols(
    api_key: str | None = None,
    secret_key: str | None = None,
    exchanges: tuple[str, ...] = MAJOR_STOCK_EXCHANGES,
) -> list[str]:
    """All currently-tradable common stocks on major exchanges.

    ARCA/BATS are excluded because they're predominantly ETFs, and a momentum
    strategy ranking ETFs alongside single stocks is a different (and much more
    correlated) strategy than the one being tested.
    """
    api_key = api_key or os.environ["ALPACA_API_KEY"]
    secret_key = secret_key or os.environ["ALPACA_SECRET_KEY"]
    client = TradingClient(api_key, secret_key, paper=True)
    assets = cast(
        list[Any],
        client.get_all_assets(
            GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
        ),
    )
    return sorted(a.symbol for a in assets if a.tradable and str(a.exchange) in exchanges)


def screen_at_date(
    client: StockHistoricalDataClient,
    symbols: list[str],
    screen_date: date,
    config: ScreenConfig | None = None,
    progress: bool = False,
) -> list[str]:
    """Symbols meeting price/liquidity floors over the window ending at `screen_date`."""
    config = config or ScreenConfig()
    start = screen_date - timedelta(days=config.window_days)
    panel = fetch_daily_bars(client, symbols, start, screen_date, progress=progress)
    if panel.empty:
        return []

    panel = panel.assign(dollar_volume=panel["close"] * panel["volume"])
    stats = panel.groupby("symbol").agg(
        median_price=("close", "median"),
        median_dollar_volume=("dollar_volume", "median"),
        n_bars=("close", "size"),
    )

    # Require most of the window to be present — a name with 3 prints in 90 days
    # can clear a median filter while being untradable in practice.
    min_bars = max(20, int(config.window_days * 0.4))
    qualified = stats[
        (stats["median_price"] >= config.min_price)
        & (stats["median_dollar_volume"] >= config.min_median_dollar_volume)
        & (stats["n_bars"] >= min_bars)
    ]
    ranked = qualified.sort_values("median_dollar_volume", ascending=False)
    return list(ranked.head(config.max_symbols_per_screen).index)


def build_union_universe(
    client: StockHistoricalDataClient,
    candidate_symbols: list[str],
    screen_dates: list[date],
    config: ScreenConfig | None = None,
    progress: bool = False,
) -> list[str]:
    universe: set[str] = set()
    for screen_date in screen_dates:
        selected = screen_at_date(client, candidate_symbols, screen_date, config, progress=progress)
        if progress:
            print(f"  screen {screen_date}: {len(selected)} symbols qualified")
        universe.update(selected)
    return sorted(universe)
