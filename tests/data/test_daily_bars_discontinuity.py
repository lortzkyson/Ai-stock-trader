from __future__ import annotations

import pandas as pd

from data.daily_bars import split_discontinuous_series


def make_series(symbol: str, prices: list[float], start: str = "2020-01-01") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(prices))
    return pd.DataFrame(
        [
            {"symbol": symbol, "timestamp": t, "open": p, "high": p, "low": p,
             "close": p, "volume": 1000}
            for t, p in zip(idx, prices)
        ]
    )


def test_bankruptcy_emergence_history_is_dropped() -> None:
    """Old equity cancelled at pennies, new shares issued under the same ticker.
    Real example: GPOR printed +52,648% ($0.14 -> $72.95) on 2021-05-18."""
    panel = make_series("GPOR", [10.0, 5.0, 1.0, 0.5, 0.14, 72.95, 73.0, 74.0])

    result = split_discontinuous_series(panel)

    assert len(result) == 3
    assert result["close"].iloc[0] == 72.95, "pre-reorganization history must be dropped"


def test_genuine_squeeze_is_preserved() -> None:
    """A real +150% move is extreme but happens; it must not be treated as an
    artifact, or the cleaner would delete exactly the events momentum trades."""
    panel = make_series("SQZ", [10.0, 10.0, 25.0, 20.0, 18.0])

    result = split_discontinuous_series(panel)

    assert len(result) == len(panel)


def test_only_the_affected_symbol_is_truncated() -> None:
    panel = pd.concat(
        [make_series("GOOD", [10.0, 11.0, 12.0, 13.0]),
         make_series("BAD", [10.0, 0.2, 50.0, 51.0])],
        ignore_index=True,
    )

    result = split_discontinuous_series(panel)

    assert (result["symbol"] == "GOOD").sum() == 4
    assert (result["symbol"] == "BAD").sum() == 2


def test_last_discontinuity_wins_when_there_are_several() -> None:
    # Two reorganizations: only history after the most recent one is valid.
    panel = make_series("TWICE", [10.0, 0.1, 40.0, 0.3, 90.0, 91.0])

    result = split_discontinuous_series(panel)

    assert result["close"].iloc[0] == 90.0
    assert len(result) == 2


def test_clean_panel_is_unchanged() -> None:
    panel = make_series("CALM", [10.0, 10.5, 10.2, 10.8])
    result = split_discontinuous_series(panel)
    assert len(result) == len(panel)


def test_empty_panel_is_handled() -> None:
    empty = pd.DataFrame(
        columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    )
    assert split_discontinuous_series(empty).empty
