from __future__ import annotations

import pytest

from execution.client import AlpacaExecutionClient, OrderSubmissionError, RetryConfig


def test_paper_only_guard() -> None:
    with pytest.raises(ValueError, match="paper-only"):
        AlpacaExecutionClient(api_key="dummy", secret_key="dummy", paper=False)


def test_with_retry_succeeds_after_transient_failures() -> None:
    retry_config = RetryConfig(max_attempts=3, backoff_seconds=0)
    client = AlpacaExecutionClient(api_key="dummy", secret_key="dummy", retry_config=retry_config)
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("transient")
        return "ok"

    result = client._with_retry(flaky)

    assert result == "ok"
    assert calls["n"] == 3


def test_with_retry_raises_after_max_attempts() -> None:
    retry_config = RetryConfig(max_attempts=2, backoff_seconds=0)
    client = AlpacaExecutionClient(api_key="dummy", secret_key="dummy", retry_config=retry_config)

    def always_fails() -> None:
        raise ConnectionError("still down")

    with pytest.raises(OrderSubmissionError):
        client._with_retry(always_fails)
