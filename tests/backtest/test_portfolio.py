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


def _volatile_prices(n_days: int, calm_vol: float, wild_vol: float, seed: int = 0):
    """Prices that are calm for the first half, turbulent for the second."""
    rng = np.random.default_rng(seed)
    vols = [calm_vol] * (n_days // 2) + [wild_vol] * (n_days - n_days // 2)
    rets = [rng.normal(0.0005, v) for v in vols]
    px = 100 * np.cumprod(1 + np.array(rets))
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame({"AAA": px}, index=idx)


def test_vol_scaling_cuts_exposure_when_volatility_rises() -> None:
    prices = _volatile_prices(400, calm_vol=0.004, wild_vol=0.05)
    # Rebalance monthly so exposure is re-evaluated repeatedly.
    targets = {d: ["AAA"] for d in prices.index[::21][:-1]}

    result = run_portfolio_backtest(
        targets, prices, prices,
        PortfolioBacktestConfig(starting_equity=100_000, slippage_bps=0,
                                vol_target=0.12, vol_lookback_days=126),
    )

    exposure = result.exposure.dropna()
    assert len(exposure) > 4
    early, late = exposure.iloc[1], exposure.iloc[-1]
    assert late < early, f"exposure should fall in the turbulent regime ({late} vs {early})"


def test_vol_scaling_never_exceeds_max_exposure() -> None:
    # Very calm prices would imply huge leverage if uncapped.
    prices = _volatile_prices(400, calm_vol=0.0005, wild_vol=0.0005)
    targets = {d: ["AAA"] for d in prices.index[::21][:-1]}

    result = run_portfolio_backtest(
        targets, prices, prices,
        PortfolioBacktestConfig(starting_equity=100_000, slippage_bps=0,
                                vol_target=0.12, max_exposure=1.0),
    )

    assert (result.exposure.dropna() <= 1.0 + 1e-9).all()


def test_vol_target_none_is_the_unscaled_baseline() -> None:
    prices = _volatile_prices(300, calm_vol=0.01, wild_vol=0.03)
    targets = {d: ["AAA"] for d in prices.index[::21][:-1]}

    unscaled = run_portfolio_backtest(
        targets, prices, prices,
        PortfolioBacktestConfig(starting_equity=100_000, slippage_bps=0, vol_target=None),
    )

    assert (unscaled.exposure.dropna() == 1.0).all()


def test_vol_scaling_uses_only_past_returns() -> None:
    """Future prices must not change an exposure decision already made."""
    prices = _volatile_prices(300, calm_vol=0.005, wild_vol=0.03, seed=7)
    targets = {d: ["AAA"] for d in prices.index[::21][:-1]}
    cfg = PortfolioBacktestConfig(starting_equity=100_000, slippage_bps=0, vol_target=0.12)

    original = run_portfolio_backtest(targets, prices, prices, cfg)

    mutated = prices.copy()
    mutated.iloc[250:] *= 5  # violently alter the far future
    mutated_result = run_portfolio_backtest(targets, mutated, mutated, cfg)

    cutoff = prices.index[250]
    a = original.exposure[original.exposure.index < cutoff]
    b = mutated_result.exposure[mutated_result.exposure.index < cutoff]
    pd.testing.assert_series_equal(a, b)


def test_short_position_profits_when_price_falls() -> None:
    close = make_prices({"AAA": [100.0, 100.0, 80.0, 80.0]})
    open_ = make_prices({"AAA": [100.0, 100.0, 80.0, 80.0]})

    result = run_portfolio_backtest(
        {}, close, open_,
        PortfolioBacktestConfig(starting_equity=10_000, slippage_bps=0,
                                short_borrow_annual_rate=0.0),
        short_targets={close.index[0]: ["AAA"]},
    )

    shorts = result.trades[result.trades["side"] == "short"]
    assert len(shorts) == 1
    # Shorted 100 shares at 100; price fell to 80 -> +2,000.
    assert result.daily_equity.iloc[-1] == pytest.approx(12_000, rel=1e-3)


def test_short_position_loses_when_price_rises() -> None:
    close = make_prices({"AAA": [100.0, 100.0, 120.0, 120.0]})
    open_ = make_prices({"AAA": [100.0, 100.0, 120.0, 120.0]})

    result = run_portfolio_backtest(
        {}, close, open_,
        PortfolioBacktestConfig(starting_equity=10_000, slippage_bps=0,
                                short_borrow_annual_rate=0.0),
        short_targets={close.index[0]: ["AAA"]},
    )

    assert result.daily_equity.iloc[-1] == pytest.approx(8_000, rel=1e-3)


def test_borrow_cost_is_charged_on_shorts() -> None:
    close = make_prices({"AAA": [100.0] * 30})
    open_ = make_prices({"AAA": [100.0] * 30})
    args = ({}, close, open_)

    free = run_portfolio_backtest(
        *args,
        PortfolioBacktestConfig(starting_equity=10_000, slippage_bps=0,
                                short_borrow_annual_rate=0.0),
        short_targets={close.index[0]: ["AAA"]},
    )
    costly = run_portfolio_backtest(
        *args,
        PortfolioBacktestConfig(starting_equity=10_000, slippage_bps=0,
                                short_borrow_annual_rate=0.50),
        short_targets={close.index[0]: ["AAA"]},
    )

    # Flat price: the only difference between the runs is borrow.
    assert costly.daily_equity.iloc[-1] < free.daily_equity.iloc[-1]


def test_short_is_covered_when_it_leaves_the_target() -> None:
    close = make_prices({"AAA": [100.0] * 6})
    open_ = make_prices({"AAA": [100.0] * 6})

    result = run_portfolio_backtest(
        {}, close, open_,
        PortfolioBacktestConfig(starting_equity=10_000, slippage_bps=0,
                                short_borrow_annual_rate=0.0),
        short_targets={close.index[0]: ["AAA"], close.index[2]: []},
    )

    covers = result.trades[result.trades["side"] == "cover"]
    assert len(covers) == 1


def test_slippage_is_adverse_on_short_and_cover() -> None:
    close = make_prices({"AAA": [100.0] * 6})
    open_ = make_prices({"AAA": [100.0] * 6})

    result = run_portfolio_backtest(
        {}, close, open_,
        PortfolioBacktestConfig(starting_equity=10_000, slippage_bps=100,
                                short_borrow_annual_rate=0.0),
        short_targets={close.index[0]: ["AAA"], close.index[2]: []},
    )

    short = result.trades[result.trades["side"] == "short"].iloc[0]
    cover = result.trades[result.trades["side"] == "cover"].iloc[0]
    assert short["price"] == pytest.approx(99.0)   # sold lower
    assert cover["price"] == pytest.approx(101.0)  # bought back higher


def test_long_short_holds_both_legs_simultaneously() -> None:
    close = make_prices({"WIN": [100.0] * 5, "LOSE": [50.0] * 5})
    open_ = make_prices({"WIN": [100.0] * 5, "LOSE": [50.0] * 5})

    result = run_portfolio_backtest(
        {close.index[0]: ["WIN"]}, close, open_,
        PortfolioBacktestConfig(starting_equity=10_000, slippage_bps=0,
                                short_borrow_annual_rate=0.0),
        short_targets={close.index[0]: ["LOSE"]},
    )

    sides = set(result.trades["side"])
    assert "buy" in sides and "short" in sides


def test_losing_short_is_trimmed_not_ratcheted_up() -> None:
    """Regression: a short that moves against you grows in absolute size while
    equity shrinks. If the rebalance only ever *adds* to shorts, the position
    ratchets up every month and the portfolio blows up (observed: 425%
    annualized vol, -97% drawdown). It must be trimmed back to target weight."""
    # AAA climbs steadily -- our short loses the whole way.
    prices = make_prices({"AAA": [10.0, 10.0, 20.0, 20.0, 40.0, 40.0, 80.0, 80.0]})
    rebalances = [prices.index[i] for i in (0, 2, 4, 6)]

    result = run_portfolio_backtest(
        {}, prices, prices,
        PortfolioBacktestConfig(starting_equity=10_000, slippage_bps=0,
                                short_borrow_annual_rate=0.0),
        short_targets={d: ["AAA"] for d in rebalances},
    )

    assert not result.trades.empty
    covers = result.trades[result.trades["side"] == "cover"]
    assert len(covers) > 0, "an oversized losing short must be trimmed back"

    # Short notional must never run away past the equity backing it.
    final_equity = result.daily_equity.iloc[-1]
    assert final_equity > 0, "portfolio should not be wiped out by a single short"


def test_short_notional_stays_near_target_weight_across_rebalances() -> None:
    prices = make_prices({"AAA": [10.0, 10.0, 15.0, 15.0, 22.0, 22.0]})
    rebalances = [prices.index[i] for i in (0, 2, 4)]

    result = run_portfolio_backtest(
        {}, prices, prices,
        PortfolioBacktestConfig(starting_equity=10_000, slippage_bps=0,
                                short_borrow_annual_rate=0.0),
        short_targets={d: ["AAA"] for d in rebalances},
    )

    # Reconstruct share count over time from the trade log.
    shares = 0
    for _, t in result.trades.iterrows():
        if t["side"] == "short":
            shares -= t["shares"]
        elif t["side"] == "cover":
            shares += t["shares"]
    # After trimming, the surviving short must be a sane size, not a runaway.
    assert abs(shares) * 22.0 < 10_000 * 3


def test_no_targets_leaves_equity_flat() -> None:
    close = make_prices({"AAA": [100.0, 110.0, 120.0]})
    open_ = make_prices({"AAA": [100.0, 110.0, 120.0]})

    result = run_portfolio_backtest(
        {}, close, open_, PortfolioBacktestConfig(starting_equity=10_000)
    )

    assert (result.daily_equity == 10_000).all()
    assert result.trades.empty
