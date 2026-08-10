from __future__ import annotations

from pathlib import Path

import pytest

from execution.structured_logging import StructuredLogger
from monitoring.performance_report import (
    DivergenceFlag,
    build_daily_equity_from_trades,
    build_live_metrics,
    compare_to_baseline,
    load_fills,
    reconstruct_trades_from_fills,
)


def _seed_log(tmp_path: Path) -> Path:
    log_path = tmp_path / "log.jsonl"
    logger = StructuredLogger(log_path=log_path)
    logger.log_signal("AAPL", proba=0.7, predicted=1)  # non-fill events should be ignored
    logger.log_fill("AAPL", side="buy", qty=10, price=100.0, order_id="o1")
    logger.log_fill("AAPL", side="sell", qty=10, price=102.0, order_id="o2")
    logger.log_fill("MSFT", side="buy", qty=5, price=300.0, order_id="o3")
    logger.log_fill("MSFT", side="sell", qty=5, price=295.0, order_id="o4")
    return log_path


def test_load_fills_ignores_non_fill_events(tmp_path: Path) -> None:
    log_path = _seed_log(tmp_path)
    fills = load_fills(log_path)
    assert len(fills) == 4
    assert set(fills["event_type"]) == {"fill"}


def test_reconstruct_trades_pairs_buys_and_sells_fifo(tmp_path: Path) -> None:
    fills = load_fills(_seed_log(tmp_path))
    trades = reconstruct_trades_from_fills(fills)

    assert len(trades) == 2
    aapl = trades[trades["symbol"] == "AAPL"].iloc[0]
    assert aapl["pnl"] == pytest.approx((102.0 - 100.0) * 10)
    msft = trades[trades["symbol"] == "MSFT"].iloc[0]
    assert msft["pnl"] == pytest.approx((295.0 - 300.0) * 5)


def test_reconstruct_trades_splits_partial_fills(tmp_path: Path) -> None:
    log_path = tmp_path / "log.jsonl"
    logger = StructuredLogger(log_path=log_path)
    logger.log_fill("AAPL", side="buy", qty=10, price=100.0, order_id="o1")
    logger.log_fill("AAPL", side="sell", qty=6, price=101.0, order_id="o2")
    logger.log_fill("AAPL", side="sell", qty=4, price=99.0, order_id="o3")

    trades = reconstruct_trades_from_fills(load_fills(log_path))

    assert len(trades) == 2
    assert trades.iloc[0]["shares"] == 6
    assert trades.iloc[1]["shares"] == 4


def test_build_daily_equity_from_trades_compounds_pnl_by_day(tmp_path: Path) -> None:
    fills = load_fills(_seed_log(tmp_path))
    trades = reconstruct_trades_from_fills(fills)

    equity = build_daily_equity_from_trades(trades, starting_equity=10_000)

    assert equity.iloc[-1] == pytest.approx(10_000 + trades["pnl"].sum())


def test_compare_to_baseline_flags_large_relative_gap() -> None:
    live = {"win_rate": 0.15, "expectancy": 0.001, "sharpe": 0.1, "max_drawdown": -0.05}
    baseline = {"win_rate": 0.4, "expectancy": 0.002, "sharpe": 1.0, "max_drawdown": -0.05}

    flags = compare_to_baseline(live, baseline, tolerance=0.5)

    flagged_metrics = {f.metric for f in flags}
    assert "win_rate" in flagged_metrics
    assert "sharpe" in flagged_metrics
    assert "max_drawdown" not in flagged_metrics  # identical, no gap
    assert all(isinstance(f, DivergenceFlag) for f in flags)


def test_compare_to_baseline_no_flags_when_close() -> None:
    live = {"win_rate": 0.40, "expectancy": 0.002, "sharpe": 1.0, "max_drawdown": -0.05}
    baseline = {"win_rate": 0.41, "expectancy": 0.0021, "sharpe": 1.05, "max_drawdown": -0.051}

    flags = compare_to_baseline(live, baseline, tolerance=0.5)

    assert flags == []


def test_build_live_metrics_end_to_end(tmp_path: Path) -> None:
    log_path = _seed_log(tmp_path)
    metrics = build_live_metrics(log_path, starting_equity=10_000)
    assert metrics["n_trades"] == 2
