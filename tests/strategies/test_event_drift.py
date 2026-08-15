from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.event_drift import (
    EventDriftConfig,
    build_drift_targets,
    detect_events,
    forward_return_study,
    summarize_events,
)


def make_panel(symbol: str, returns: list[float], volumes: list[float] | None = None,
               start: str = "2020-01-01") -> pd.DataFrame:
    prices = 100 * np.cumprod(1 + np.array(returns))
    idx = pd.bdate_range(start, periods=len(prices))
    vols = volumes if volumes is not None else [1_000_000.0] * len(prices)
    return pd.DataFrame(
        [{"symbol": symbol, "timestamp": t, "open": p, "high": p * 1.01,
          "low": p * 0.99, "close": p, "volume": v}
         for t, p, v in zip(idx, prices, vols)]
    )


def test_event_requires_both_abnormal_return_and_abnormal_volume() -> None:
    rng = np.random.default_rng(0)
    n = 200
    rets = list(rng.normal(0, 0.01, n))
    vols = [1_000_000.0] * n

    # Big move on NORMAL volume -> not an event.
    rets[150] = 0.10
    # Normal move on HUGE volume -> not an event.
    vols[160] = 10_000_000.0
    # Big move on huge volume -> event.
    rets[170] = 0.10
    vols[170] = 10_000_000.0

    panel = make_panel("AAA", rets, vols)
    result = detect_events(panel, EventDriftConfig())

    assert not result.loc[150, "is_event"]
    assert not result.loc[160, "is_event"]
    assert result.loc[170, "is_event"]


def test_event_threshold_is_relative_to_each_symbols_own_volatility() -> None:
    """A 5% day is routine for a volatile name and extraordinary for a calm one.
    An absolute threshold would flag only high-beta names."""
    rng = np.random.default_rng(1)
    n = 200
    calm = list(rng.normal(0, 0.002, n))
    wild = list(rng.normal(0, 0.05, n))
    calm[170] = 0.05
    wild[170] = 0.05
    vols = [1_000_000.0] * n
    vols[170] = 10_000_000.0

    calm_events = detect_events(make_panel("CALM", calm, vols), EventDriftConfig())
    wild_events = detect_events(make_panel("WILD", wild, vols), EventDriftConfig())

    assert calm_events.loc[170, "is_event"], "5% should be an event for a calm name"
    assert not wild_events.loc[170, "is_event"], "5% is unremarkable for a volatile name"


def test_detection_uses_only_trailing_windows() -> None:
    """An event must be identifiable at its own close — the thresholds it clears
    cannot be computed using the event bar itself or anything after it."""
    rng = np.random.default_rng(3)
    n = 200
    rets = list(rng.normal(0, 0.01, n))
    vols = [1_000_000.0] * n
    rets[120] = 0.08
    vols[120] = 5_000_000.0

    original = detect_events(make_panel("AAA", rets, vols), EventDriftConfig())

    mutated_rets = rets[:150] + list(rng.normal(0, 0.5, n - 150))
    mutated = detect_events(make_panel("AAA", mutated_rets, vols), EventDriftConfig())

    pd.testing.assert_series_equal(
        original["is_event"].iloc[:150], mutated["is_event"].iloc[:150], check_names=False
    )


def test_positive_and_negative_events_are_separated() -> None:
    rng = np.random.default_rng(2)
    n = 200
    rets = list(rng.normal(0, 0.01, n))
    vols = [1_000_000.0] * n
    rets[150] = 0.10
    vols[150] = 10_000_000.0
    rets[170] = -0.10
    vols[170] = 10_000_000.0

    panel = make_panel("AAA", rets, vols)
    summary = summarize_events(panel, EventDriftConfig())

    assert summary["positive_events"] >= 1
    assert summary["negative_events"] >= 1


def test_drift_targets_hold_names_after_an_event() -> None:
    rng = np.random.default_rng(4)
    n = 300
    rets = list(rng.normal(0, 0.01, n))
    vols = [1_000_000.0] * n
    rets[100] = 0.12
    vols[100] = 10_000_000.0

    panel = make_panel("AAA", rets, vols)
    targets = build_drift_targets(panel, EventDriftConfig(holding_days=40))

    assert any("AAA" in syms for syms in targets.values())


def test_no_events_yields_no_targets() -> None:
    panel = make_panel("CALM", [0.001] * 200)
    assert build_drift_targets(panel, EventDriftConfig()) == {}


def test_forward_return_study_reports_all_horizons() -> None:
    rng = np.random.default_rng(5)
    n = 400
    rets = list(rng.normal(0.0004, 0.012, n))
    vols = [1_000_000.0] * n
    for i in (120, 200, 260):
        rets[i] = 0.09
        vols[i] = 8_000_000.0

    study = forward_return_study(make_panel("AAA", rets, vols), EventDriftConfig())

    assert list(study["horizon_days"]) == [5, 10, 20, 40, 60]
    assert "positive_edge_vs_baseline" in study.columns
