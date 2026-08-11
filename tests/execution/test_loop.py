from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from execution import kill_switch
from execution.loop import run_iteration
from execution.position_state import load_open_positions, save_open_positions
from risk.engine import RiskConfig
from risk.state_persistence import load_risk_state, save_risk_state

from .conftest import make_clean_bars


@dataclass
class FakePosition:
    symbol: str
    qty: str


@dataclass
class FakeOrder:
    id: str = "order-1"


@dataclass
class FakeExecClient:
    equity: float = 100_000.0
    positions: list = field(default_factory=list)
    submitted: list = field(default_factory=list)
    market_open: bool = True

    def get_account_equity(self) -> float:
        return self.equity

    def is_market_open(self) -> bool:
        return self.market_open

    def get_positions(self) -> list:
        return self.positions

    def submit_market_order(self, symbol, qty, side, client_order_id):
        order = FakeOrder(id=f"order-{len(self.submitted) + 1}")
        self.submitted.append(
            {"symbol": symbol, "qty": qty, "side": str(side), "order_id": order.id}
        )
        return order


@dataclass
class FakeBarsClient:
    bars_by_symbol: dict

    def fetch_bars(self, query):
        symbol = query.symbols[0]
        return self.bars_by_symbol.get(
            symbol, pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        )


class _AlwaysBuyModel:
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.tile([0.1, 0.9], (len(X), 1))


class _NeverBuyModel:
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.tile([0.9, 0.1], (len(X), 1))


def _paths(tmp_path: Path) -> dict:
    return {
        "risk_state_path": tmp_path / "risk_state.json",
        "position_state_path": tmp_path / "positions.json",
        "kill_switch_path": tmp_path / "KILL_SWITCH_ENGAGED",
        "log_path": tmp_path / "log.jsonl",
    }


def test_kill_switch_engaged_skips_everything(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    kill_switch.engage(paths["kill_switch_path"])

    # exec_client=None would normally construct a real client — kill switch
    # must short-circuit before that ever happens.
    run_iteration(["AAPL"], exec_client=None, bars_client=None, model=None, **paths)

    assert not paths["log_path"].exists() or "kill_switch_engaged" in paths["log_path"].read_text()


def test_market_closed_skips_everything(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    exec_client = FakeExecClient(equity=100_000.0, market_open=False)
    bars_client = FakeBarsClient(bars_by_symbol={})

    run_iteration(
        ["AAPL"], exec_client=exec_client, bars_client=bars_client,
        model=_AlwaysBuyModel(), **paths,
    )

    assert exec_client.submitted == []
    assert "market_closed" in paths["log_path"].read_text()


def test_entry_signal_submits_order_and_persists_position(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    bars = make_clean_bars(["2026-01-05"])
    exec_client = FakeExecClient(equity=100_000.0, positions=[])
    bars_client = FakeBarsClient(bars_by_symbol={"AAPL": bars})

    run_iteration(
        ["AAPL"], exec_client=exec_client, bars_client=bars_client,
        model=_AlwaysBuyModel(), risk_config=RiskConfig(), **paths,
    )

    assert len(exec_client.submitted) == 1
    assert exec_client.submitted[0]["symbol"] == "AAPL"
    assert exec_client.submitted[0]["side"] == "OrderSide.BUY"

    positions = load_open_positions(paths["position_state_path"])
    assert "AAPL" in positions
    assert positions["AAPL"]["stop_loss_price"] > 0


def test_no_signal_submits_nothing(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    bars = make_clean_bars(["2026-01-05"])
    exec_client = FakeExecClient(equity=100_000.0, positions=[])
    bars_client = FakeBarsClient(bars_by_symbol={"AAPL": bars})

    run_iteration(
        ["AAPL"], exec_client=exec_client, bars_client=bars_client,
        model=_NeverBuyModel(), risk_config=RiskConfig(), **paths,
    )

    assert exec_client.submitted == []
    assert load_open_positions(paths["position_state_path"]) == {}
    # A healthy quiet run must still leave a trace, distinguishing it from a
    # silent crash.
    assert "loop_completed" in paths["log_path"].read_text()


def test_insufficient_warmup_bars_skips_symbol(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    bars = make_clean_bars(["2026-01-05"]).iloc[:10]  # far fewer than 60-bar warmup
    exec_client = FakeExecClient(equity=100_000.0, positions=[])
    bars_client = FakeBarsClient(bars_by_symbol={"AAPL": bars})

    run_iteration(
        ["AAPL"], exec_client=exec_client, bars_client=bars_client,
        model=_AlwaysBuyModel(), **paths,
    )

    assert exec_client.submitted == []


def test_stop_loss_exit_closes_tracked_position(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    save_open_positions(
        {
            "AAPL": {
                "entry_price": 100.0,
                "stop_loss_price": 99.0,
                "take_profit_price": 102.0,
                "entry_date": pd.Timestamp.now(tz="UTC").date().isoformat(),
                "max_holding_deadline": (
                    datetime.now(timezone.utc) + timedelta(days=1)
                ).isoformat(),
            }
        },
        paths["position_state_path"],
    )

    # A bar within regular session hours whose low crosses the stop (fake
    # bars_client ignores the query's real time window, so a fixed
    # in-session timestamp keeps this test independent of wall-clock time).
    bar_ts = pd.Timestamp("2026-01-05 10:00", tz="America/New_York")
    exit_bar = pd.DataFrame(
        [{"timestamp": bar_ts, "open": 99.5, "high": 99.6, "low": 98.5, "close": 99.0,
          "volume": 1000}]
    )
    exec_client = FakeExecClient(equity=100_000.0, positions=[FakePosition("AAPL", "10")])
    bars_client = FakeBarsClient(bars_by_symbol={"AAPL": exit_bar})

    run_iteration(
        ["AAPL"], exec_client=exec_client, bars_client=bars_client,
        model=_NeverBuyModel(), **paths,
    )

    assert len(exec_client.submitted) == 1
    assert exec_client.submitted[0]["side"] == "OrderSide.SELL"
    assert load_open_positions(paths["position_state_path"]) == {}


def test_drawdown_breaker_blocks_new_entries(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    from risk.engine import RiskEngine

    engine = RiskEngine(RiskConfig(max_drawdown_pct=0.10))
    engine.drawdown_breaker.update(100_000)
    engine.drawdown_breaker.update(85_000)  # -15%, trips it
    save_risk_state(engine, paths["risk_state_path"])

    bars = make_clean_bars(["2026-01-05"])
    exec_client = FakeExecClient(equity=85_000.0, positions=[])
    bars_client = FakeBarsClient(bars_by_symbol={"AAPL": bars})

    run_iteration(
        ["AAPL"], exec_client=exec_client, bars_client=bars_client,
        model=_AlwaysBuyModel(), **paths,
    )

    assert exec_client.submitted == []


def test_risk_state_persists_across_iterations(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    bars = make_clean_bars(["2026-01-05"])
    exec_client = FakeExecClient(equity=100_000.0, positions=[])
    bars_client = FakeBarsClient(bars_by_symbol={"AAPL": bars})

    run_iteration(
        ["AAPL"], exec_client=exec_client, bars_client=bars_client,
        model=_NeverBuyModel(), **paths,
    )

    from risk.engine import RiskConfig as RC
    from risk.engine import RiskEngine as RE

    restored = RE(RC())
    load_risk_state(restored, paths["risk_state_path"])
    assert restored.drawdown_breaker._peak_equity == 100_000.0
