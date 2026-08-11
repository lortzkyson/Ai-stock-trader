from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.portfolio import PortfolioBacktestConfig, run_portfolio_backtest


def make_prices(data: dict[str, list[float]], start: str = "2026-01-01") -> pd.DataFrame:
    n = len(next(iter(data.values())))
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame(data, index=idx)


def test_fills_at_next_open_not_signal_date() -> None:
    # Signal on day 0; the fill must use day 1's open (200), not day 0's.
    close = make_prices({"AAA": [100.0, 100.0, 100.0]})
    open_ = make_prices({"AAA": [100.0, 200.0, 200.0]})
    targets = {close.index[0]: ["AAA"]}

    result = run_portfolio_backtest(
        targets, close, open_, PortfolioBacktestConfig(starting_equity=10_000, slippage_bps=0)
    )

    buys = result.trades[result.trades["side"] == "buy"]
    assert len(buys) == 1
    assert buys.iloc[0]["price"] == pytest.approx(200.0)
    assert buys.iloc[0]["date"] == close.index[1]


def test_equity_is_conserved_through_rebalance_minus_costs() -> None:
    """The classic portfolio-accounting bug is creating or destroying money at
    rebalance. With flat prices and no slippage, equity should only fall by fees."""
    close = make_prices({"AAA": [100.0] * 6, "BBB": [50.0] * 6})
    open_ = make_prices({"AAA": [100.0] * 6, "BBB": [50.0] * 6})
    targets = {close.index[0]: ["AAA"], close.index[2]: ["BBB"]}

    result = run_portfolio_backtest(
        targets, close, open_, PortfolioBacktestConfig(starting_equity=10_000, slippage_bps=0)
    )

    total_fees = result.trades["fees"].sum()
    final_equity = result.daily_equity.iloc[-1]
    # Whole-share rounding leaves a little uninvested cash, so allow a small band,
    # but nothing should create money.
    assert final_equity <= 10_000
    assert final_equity >= 10_000 - total_fees - 200


def test_slippage_is_adverse_in_both_directions() -> None:
    close = make_prices({"AAA": [100.0] * 5})
    open_ = make_prices({"AAA": [100.0] * 5})
    targets = {close.index[0]: ["AAA"], close.index[2]: []}

    result = run_portfolio_backtest(
        targets, close, open_, PortfolioBacktestConfig(starting_equity=10_000, slippage_bps=100)
    )

    buys = result.trades[result.trades["side"] == "buy"]
    sells = result.trades[result.trades["side"] == "sell"]
    assert buys.iloc[0]["price"] == pytest.approx(101.0)  # paid up
    assert sells.iloc[0]["price"] == pytest.approx(99.0)  # sold down


def test_positions_are_equal_weighted() -> None:
    close = make_prices({"AAA": [100.0] * 4, "BBB": [25.0] * 4})
    open_ = make_prices({"AAA": [100.0] * 4, "BBB": [25.0] * 4})
    targets = {close.index[0]: ["AAA", "BBB"]}

    result = run_portfolio_backtest(
        targets, close, open_, PortfolioBacktestConfig(starting_equity=10_000, slippage_bps=0)
    )

    buys = result.trades.set_index("symbol")
    # $5,000 each -> 50 shares of AAA @100, 200 shares of BBB @25
    assert buys.loc["AAA", "shares"] == 50
    assert buys.loc["BBB", "shares"] == 200


def test_sell_fees_are_charged() -> None:
    close = make_prices({"AAA": [100.0] * 5})
    open_ = make_prices({"AAA": [100.0] * 5})
    targets = {close.index[0]: ["AAA"], close.index[2]: []}

    result = run_portfolio_backtest(
        targets, close, open_, PortfolioBacktestConfig(starting_equity=10_000, slippage_bps=0)
    )

    sells = result.trades[result.trades["side"] == "sell"]
    assert len(sells) == 1
    assert sells.iloc[0]["fees"] > 0


def test_position_with_no_price_is_dropped_not_silently_held() -> None:
    close = make_prices({"AAA": [100.0, 100.0, np.nan, np.nan], "BBB": [50.0] * 4})
    open_ = make_prices({"AAA": [100.0, 100.0, np.nan, np.nan], "BBB": [50.0] * 4})
    targets = {close.index[0]: ["AAA"], close.index[2]: ["BBB"]}

    result = run_portfolio_backtest(
        targets, close, open_, PortfolioBacktestConfig(starting_equity=10_000, slippage_bps=0)
    )

    drops = result.trades[result.trades["side"] == "drop_no_price"]
    assert len(drops) == 1
    assert drops.iloc[0]["symbol"] == "AAA"


def test_equity_tracks_price_appreciation_while_held() -> None:
    close = make_prices({"AAA": [100.0, 100.0, 110.0, 120.0]})
    open_ = make_prices({"AAA": [100.0, 100.0, 110.0, 120.0]})
    targets = {close.index[0]: ["AAA"]}

    result = run_portfolio_backtest(
        targets, close, open_, PortfolioBacktestConfig(starting_equity=10_000, slippage_bps=0)
    )

    # Bought 100 shares at 100 on day 1; by day 3 they're worth 120 each.
    assert result.daily_equity.iloc[-1] == pytest.approx(12_000, rel=1e-3)


def test_no_targets_leaves_equity_flat() -> None:
    close = make_prices({"AAA": [100.0, 110.0, 120.0]})
    open_ = make_prices({"AAA": [100.0, 110.0, 120.0]})

    result = run_portfolio_backtest(
        {}, close, open_, PortfolioBacktestConfig(starting_equity=10_000)
    )

    assert (result.daily_equity == 10_000).all()
    assert result.trades.empty
