"""Daily-bar fetching, including the alias-collision guard."""

from __future__ import annotations

from datetime import date

import pandas as pd

import data.daily_bars as db
from data.daily_bars import check_daily_quality, clean_daily_panel, fetch_daily_bars_verified


def _rows(symbol: str, n: int = 3) -> list[dict]:
    return [
        {
            "symbol": symbol,
            "timestamp": pd.Timestamp("2020-06-10", tz="UTC") + pd.Timedelta(days=i),
            "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "volume": 1000,
        }
        for i in range(n)
    ]


def test_alias_collision_symbol_is_recovered_on_retry(monkeypatch) -> None:
    """Regression guard for a real Alpaca behaviour.

    FRC and FRCB are the same entity (First Republic's NYSE listing and its
    post-failure OTC ticker). Requested together, Alpaca returns only FRCB and
    silently drops FRC — indistinguishable from "FRC has no data". This cost a
    genuinely liquid $124M/day name its place in the universe.
    """
    calls: list[list[str]] = []

    def fake_fetch(client, symbols, start, end, batch_size=500, progress=False):  # noqa: ANN001
        calls.append(list(symbols))
        out: list[dict] = []
        for s in symbols:
            # The collision: FRC yields nothing when FRCB is in the same request.
            if s == "FRC" and "FRCB" in symbols:
                continue
            if s in {"FRC", "FRCB", "AAPL"}:
                out.extend(_rows(s))
        return pd.DataFrame(out)

    monkeypatch.setattr(db, "fetch_daily_bars", fake_fetch)

    panel = fetch_daily_bars_verified(
        None, ["AAPL", "FRC", "FRCB"], date(2020, 6, 1), date(2020, 6, 30)
    )

    assert set(panel["symbol"]) == {"AAPL", "FRC", "FRCB"}, "FRC must survive the retry pass"
    assert len(calls) == 2, "should have made a verification pass"
    assert calls[1] == ["FRC"], "retry should request only the dropped symbol"


def test_no_retry_when_nothing_missing(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_fetch(client, symbols, start, end, batch_size=500, progress=False):  # noqa: ANN001
        calls.append(list(symbols))
        return pd.DataFrame([r for s in symbols for r in _rows(s)])

    monkeypatch.setattr(db, "fetch_daily_bars", fake_fetch)
    fetch_daily_bars_verified(None, ["AAPL", "MSFT"], date(2020, 6, 1), date(2020, 6, 30))
    assert len(calls) == 1


def test_genuinely_absent_symbols_stay_absent(monkeypatch) -> None:
    def fake_fetch(client, symbols, start, end, batch_size=500, progress=False):  # noqa: ANN001
        return pd.DataFrame([r for s in symbols if s == "AAPL" for r in _rows(s)])

    monkeypatch.setattr(db, "fetch_daily_bars", fake_fetch)
    panel = fetch_daily_bars_verified(
        None, ["AAPL", "NOPE"], date(2020, 6, 1), date(2020, 6, 30)
    )
    assert set(panel["symbol"]) == {"AAPL"}


def test_quality_report_and_cleaning() -> None:
    panel = pd.DataFrame(_rows("AAPL") + _rows("MSFT"))
    panel.loc[0, "close"] = -1.0  # impossible
    panel = pd.concat([panel, panel.iloc[[1]]], ignore_index=True)  # duplicate

    report = check_daily_quality(panel)
    assert report.impossible_value_rows == 1
    assert report.duplicate_rows == 1
    assert not report.is_clean

    cleaned = clean_daily_panel(panel)
    assert (cleaned["close"] > 0).all()
    assert not cleaned.duplicated(subset=["symbol", "timestamp"]).any()
