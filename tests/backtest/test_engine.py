from __future__ import annotations

import pandas as pd
import pytest

from backtest.engine import run_backtest
from backtest.fills import FillConfig
from features.labeling import TripleBarrierConfig
from risk.engine import RiskConfig, RiskEngine
from risk.position_sizing import fixed_fractional_shares


def _bars(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp(ts, tz="America/New_York"),
                "open": o,
                "high": h,
                "low": low,
                "close": c,
                "volume": 1_000,
            }
            for ts, o, h, low, c in rows
        ]
    )


def _signals(bars: pd.DataFrame, signal_indices: set[int]) -> pd.Series:
    return pd.Series(
        [1 if i in signal_indices else 0 for i in range(len(bars))], index=bars.index
    )


def test_entry_fills_next_bar_and_exits_on_take_profit() -> None:
    bars = _bars(
        [
            ("2026-01-05 09:30", 99.9, 100.0, 99.8, 99.9),  # signal here
            ("2026-01-05 09:31", 100.0, 100.3, 99.9, 100.1),  # entry fills at this open (100.0)
            ("2026-01-05 09:32", 100.2, 100.5, 100.0, 100.3),
            ("2026-01-05 09:33", 100.4, 102.5, 100.1, 102.3),  # high crosses target (102.0)
        ]
    )
    signals = _signals(bars, {0})
    barrier = TripleBarrierConfig(profit_target_pct=0.02, stop_loss_pct=0.01, max_holding_bars=10)
    risk_config = RiskConfig(barrier_config=barrier, daily_loss_limit_pct=0.5, max_drawdown_pct=0.5)
    engine = RiskEngine(risk_config)
    fill_config = FillConfig(slippage_bps=0.0)
    calendar = [pd.Timestamp("2026-01-05").date()]

    result = run_backtest(
        {"AAA": bars}, {"AAA": signals}, engine, fill_config,
        starting_equity=100_000, trading_calendar=calendar,
    )

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["entry_price"] == pytest.approx(100.0)
    assert trade["exit_price"] == pytest.approx(102.0)  # target, no slippage
    assert trade["exit_reason"] == "take_profit"
    assert trade["pnl"] > 0

    expected_shares = fixed_fractional_shares(
        equity=100_000, entry_price=100.0, stop_loss_pct=0.01,
        risk_per_trade_pct=risk_config.risk_per_trade_pct,
        max_position_fraction=risk_config.max_position_fraction,
    )
    assert trade["shares"] == expected_shares


def test_stop_loss_wins_tie_when_both_barriers_touched_same_bar() -> None:
    bars = _bars(
        [
            ("2026-01-05 09:30", 99.9, 100.0, 99.8, 99.9),
            ("2026-01-05 09:31", 100.0, 100.1, 99.9, 100.0),  # entry at open=100.0
            ("2026-01-05 09:32", 100.0, 103.0, 98.0, 100.5),  # both target and stop touched
        ]
    )
    signals = _signals(bars, {0})
    barrier = TripleBarrierConfig(profit_target_pct=0.02, stop_loss_pct=0.01, max_holding_bars=10)
    risk_config = RiskConfig(barrier_config=barrier, daily_loss_limit_pct=0.5, max_drawdown_pct=0.5)
    engine = RiskEngine(risk_config)
    fill_config = FillConfig(slippage_bps=0.0)
    calendar = [pd.Timestamp("2026-01-05").date()]

    result = run_backtest(
        {"AAA": bars}, {"AAA": signals}, engine, fill_config,
        starting_equity=100_000, trading_calendar=calendar,
    )

    assert len(result.trades) == 1
    assert result.trades.iloc[0]["exit_reason"] == "stop_loss"


def test_max_holding_exit_when_no_barrier_touched() -> None:
    rows = [("2026-01-05 09:30", 99.9, 100.0, 99.8, 99.9)]  # signal bar
    for i in range(1, 6):
        rows.append((f"2026-01-05 09:{30+i}", 100.0, 100.05, 99.95, 100.0))
    bars = _bars(rows)
    signals = _signals(bars, {0})
    barrier = TripleBarrierConfig(profit_target_pct=0.5, stop_loss_pct=0.5, max_holding_bars=3)
    risk_config = RiskConfig(barrier_config=barrier, daily_loss_limit_pct=0.5, max_drawdown_pct=0.5)
    engine = RiskEngine(risk_config)
    fill_config = FillConfig(slippage_bps=0.0)
    calendar = [pd.Timestamp("2026-01-05").date()]

    result = run_backtest(
        {"AAA": bars}, {"AAA": signals}, engine, fill_config,
        starting_equity=100_000, trading_calendar=calendar,
    )

    assert len(result.trades) == 1
    assert result.trades.iloc[0]["exit_reason"] == "max_holding"
    assert result.trades.iloc[0]["holding_bars"] == 3


def test_pdt_limit_blocks_a_fourth_same_day_round_trip_under_25k() -> None:
    rows = []
    for day in range(4):
        date_str = f"2026-01-0{5 + day}"
        rows.append((f"{date_str} 09:30", 100.0, 100.1, 99.9, 100.0))  # signal
        rows.append((f"{date_str} 09:31", 100.0, 100.1, 99.9, 100.0))  # entry at open=100
        rows.append((f"{date_str} 09:32", 100.0, 103.0, 99.9, 102.5))  # target hit -> day trade
    bars = _bars(rows)
    signal_indices = {0, 3, 6, 9}
    signals = _signals(bars, signal_indices)

    barrier = TripleBarrierConfig(profit_target_pct=0.02, stop_loss_pct=0.01, max_holding_bars=10)
    risk_config = RiskConfig(
        barrier_config=barrier, daily_loss_limit_pct=0.5, max_drawdown_pct=0.5,
        pdt_equity_threshold=25_000.0,
    )
    engine = RiskEngine(risk_config)
    fill_config = FillConfig(slippage_bps=0.0)
    calendar = [pd.Timestamp(f"2026-01-0{5 + d}").date() for d in range(4)]

    result = run_backtest(
        {"AAA": bars}, {"AAA": signals}, engine, fill_config,
        starting_equity=10_000, trading_calendar=calendar,
    )

    # Only 3 day trades should have executed; the 4th day's signal is blocked by PDT.
    assert len(result.trades) == 3
    assert all(result.trades["is_day_trade"])


def test_fees_are_deducted_from_realized_pnl() -> None:
    bars = _bars(
        [
            ("2026-01-05 09:30", 99.9, 100.0, 99.8, 99.9),
            ("2026-01-05 09:31", 100.0, 100.3, 99.9, 100.1),
            ("2026-01-05 09:32", 100.2, 100.5, 100.0, 100.3),
            ("2026-01-05 09:33", 100.4, 102.5, 100.1, 102.3),
        ]
    )
    signals = _signals(bars, {0})
    barrier = TripleBarrierConfig(profit_target_pct=0.02, stop_loss_pct=0.01, max_holding_bars=10)
    risk_config = RiskConfig(barrier_config=barrier, daily_loss_limit_pct=0.5, max_drawdown_pct=0.5)
    engine = RiskEngine(risk_config)
    fill_config = FillConfig(slippage_bps=0.0)
    calendar = [pd.Timestamp("2026-01-05").date()]

    result = run_backtest(
        {"AAA": bars}, {"AAA": signals}, engine, fill_config,
        starting_equity=100_000, trading_calendar=calendar,
    )

    trade = result.trades.iloc[0]
    gross = (trade["exit_price"] - trade["entry_price"]) * trade["shares"]
    assert trade["fees"] > 0
    assert trade["pnl"] == pytest.approx(gross - trade["fees"])
