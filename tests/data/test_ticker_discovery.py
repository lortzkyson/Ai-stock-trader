from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from data.ticker_discovery import (
    all_candidate_tickers,
    discover_master_universe,
    probe_existing_symbols,
    union_of,
)


class FakeClient:
    """Returns bars only for symbols in `existing`, mimicking Alpaca's behaviour
    of silently omitting symbols with no data rather than erroring."""

    def __init__(self, existing: set[str]) -> None:
        self.existing = existing
        self.requests: list[list[str]] = []

    def get_stock_bars(self, request):  # noqa: ANN001
        symbols = list(request.symbol_or_symbols)
        self.requests.append(symbols)
        rows = [
            {
                "symbol": s, "timestamp": pd.Timestamp("2020-06-10", tz="UTC"),
                "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "volume": 1000,
            }
            for s in symbols
            if s in self.existing
        ]
        return _FakeBarSet(pd.DataFrame(rows))


class _FakeBarSet:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    @property
    def df(self) -> pd.DataFrame:
        if self._df.empty:
            return self._df
        return self._df.set_index(["symbol", "timestamp"])


def test_candidate_space_sizes() -> None:
    assert len(all_candidate_tickers(1)) == 26
    assert len(all_candidate_tickers(2)) == 26 + 26**2
    assert len(all_candidate_tickers(4)) == 26 + 26**2 + 26**3 + 26**4


def test_candidate_space_is_uppercase_letters_only() -> None:
    sample = all_candidate_tickers(2)
    assert "A" in sample and "AA" in sample
    assert all(s.isalpha() and s.isupper() for s in sample)


def test_probe_returns_only_symbols_with_data(tmp_path: Path, monkeypatch) -> None:
    import data.ticker_discovery as td

    client = FakeClient({"AAPL", "MSFT"})
    # Patch fetch_daily_bars to avoid the BarSet isinstance check against a fake.
    monkeypatch.setattr(
        td, "fetch_daily_bars",
        lambda c, syms, s, e, batch_size=500, progress=False: pd.DataFrame(
            [{"symbol": x, "timestamp": pd.Timestamp("2020-06-10", tz="UTC"),
              "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1}
             for x in syms if x in client.existing]
        ),
    )

    found = probe_existing_symbols(
        client, ["AAPL", "MSFT", "ZZZZ"], date(2020, 6, 15), cache_dir=tmp_path
    )

    assert found == {"AAPL", "MSFT"}


def test_probe_result_is_cached_and_reused(tmp_path: Path, monkeypatch) -> None:
    import data.ticker_discovery as td

    calls = {"n": 0}

    def fake_fetch(c, syms, s, e, batch_size=500, progress=False):  # noqa: ANN001
        calls["n"] += 1
        return pd.DataFrame(
            [{"symbol": "AAPL", "timestamp": pd.Timestamp("2020-06-10", tz="UTC"),
              "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1}]
        )

    monkeypatch.setattr(td, "fetch_daily_bars", fake_fetch)

    first = probe_existing_symbols(None, ["AAPL"], date(2020, 6, 15), cache_dir=tmp_path)
    second = probe_existing_symbols(None, ["AAPL"], date(2020, 6, 15), cache_dir=tmp_path)

    assert first == second == {"AAPL"}
    assert calls["n"] == 1, "second call should have been served from cache"
    cached = json.loads((tmp_path / "probe_2020-06-15.json").read_text())
    assert cached == ["AAPL"]


def test_discover_master_universe_unions_across_dates(tmp_path: Path, monkeypatch) -> None:
    import data.ticker_discovery as td

    alive = {
        date(2020, 6, 15): {"OLDCO", "AAPL"},   # OLDCO later delists
        date(2024, 6, 15): {"AAPL", "NEWCO"},   # NEWCO lists later
    }

    def fake_fetch(c, syms, s, e, batch_size=500, progress=False):  # noqa: ANN001
        today = e
        present = alive[today]
        return pd.DataFrame(
            [{"symbol": x, "timestamp": pd.Timestamp("2020-06-10", tz="UTC"),
              "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1}
             for x in syms if x in present]
        )

    monkeypatch.setattr(td, "fetch_daily_bars", fake_fetch)
    monkeypatch.setattr(td, "all_candidate_tickers", lambda max_len=4: ["AAPL", "OLDCO", "NEWCO"])

    by_date = discover_master_universe(
        None, [date(2020, 6, 15), date(2024, 6, 15)], cache_dir=tmp_path
    )

    # The whole point: a delisted name present only in the earlier probe must
    # survive into the master universe.
    assert union_of(by_date) == ["AAPL", "NEWCO", "OLDCO"]
    assert "OLDCO" in by_date[date(2020, 6, 15)]
    assert "OLDCO" not in by_date[date(2024, 6, 15)]


def test_union_of_empty() -> None:
    assert union_of({}) == []
