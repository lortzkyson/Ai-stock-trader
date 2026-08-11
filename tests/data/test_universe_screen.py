"""Fund-exclusion tests.

Every name below is a real Alpaca `Asset.name` string, checked against the live
API while building this filter — not invented examples. That matters because
the whole filter is name-pattern matching, so it's only as good as the real
strings it was validated on.
"""

from __future__ import annotations

import pytest

from data.universe_screen import looks_like_fund

ETF_NAMES = [
    "iShares MSCI All Country Asia ex Japan ETF",
    "Direxion Shares ETF Trust Direxion Daily AAPL Bull",
    "ProShares Trust ProShares Ultra Nasdaq Biotechnology",
    "ProShares UltraPro QQQ",
    "WisdomTree Trust WisdomTree U.S. Quality Dividend Growth Fund",
    "First Trust NASDAQ Clean Edge Smart Grid Infrastructure Index",
    "Invesco QQQ Trust, Series 1",
    "State Street SPDR S&P 500 ETF Trust",
    "PIMCO Active Bond Exchange-Traded Fund",
    "PIMCO Dynamic Income Fund",
    "The RBB Fund, Inc. F/m US Treasury 3 Month Bill Fund",
]

OPERATING_COMPANY_NAMES = [
    "Apple Inc. Common Stock",
    "Microsoft Corporation Common Stock",
    "Agilent Technologies Inc.",
    "BERKSHIRE HATHAWAY Class B",
    "JPMorgan Chase & Co.",
    # REITs — legitimate common equity, and the reason bare "Trust" is not matched.
    "Digital Realty Trust, Inc.",
    "Essex Property Trust, Inc",
    "Federal Realty Investment Trust",
    "American Assets Trust, Inc.",
    "Arbor Realty Trust, Inc.",
    "Realty Income Corporation",
    "Simon Property Group, Inc.",
    # ADRs and foreign ordinary shares — real equities that mention "Shares".
    "Alibaba Group Holding Limited American Depositary Shares",
    "ASML Holding N.V. New York Registry Shares",
    "Arm Holdings plc American Depositary Shares",
    "Amcor plc Ordinary Shares",
    "Aurora Cannabis Inc. Common Shares",
]


@pytest.mark.parametrize("name", ETF_NAMES)
def test_funds_are_detected(name: str) -> None:
    assert looks_like_fund(name), f"should have been flagged as a fund: {name}"


@pytest.mark.parametrize("name", OPERATING_COMPANY_NAMES)
def test_operating_companies_are_not_detected(name: str) -> None:
    assert not looks_like_fund(name), f"wrongly flagged as a fund: {name}"


def test_reits_survive_because_bare_trust_is_not_matched() -> None:
    """Regression guard: adding r'\\bTrust\\b' to the pattern would silently
    delete most REITs from the universe."""
    reits = [n for n in OPERATING_COMPANY_NAMES if "Trust" in n]
    assert len(reits) >= 4
    assert not any(looks_like_fund(n) for n in reits)


def test_handles_missing_name() -> None:
    assert not looks_like_fund(None)
    assert not looks_like_fund("")
