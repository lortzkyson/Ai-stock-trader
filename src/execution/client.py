"""Alpaca order submission/cancellation wrapper with retries for transient
API failures — paper endpoint only. Live trading is out of scope for this
phase; see docs/pre-mortem.md and Phase 8's go/no-go review before anything
targets a real-money account.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar, cast

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest, MarketOrderRequest

logger = logging.getLogger("execution")

T = TypeVar("T")


class OrderSubmissionError(RuntimeError):
    pass


@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int = 3
    backoff_seconds: float = 1.0


class AlpacaExecutionClient:
    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool = True,
        retry_config: RetryConfig | None = None,
    ) -> None:
        if not paper:
            raise ValueError(
                "AlpacaExecutionClient is paper-only in Phase 7 — live trading requires "
                "the Phase 8 go/no-go review and explicit approval first."
            )
        api_key = api_key or os.environ["ALPACA_API_KEY"]
        secret_key = secret_key or os.environ["ALPACA_SECRET_KEY"]
        self._client = TradingClient(api_key, secret_key, paper=True)
        self.retry_config = retry_config or RetryConfig()

    def _with_retry(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        last_exc: Exception | None = None
        for attempt in range(1, self.retry_config.max_attempts + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # Alpaca SDK raises various types for API errors
                last_exc = exc
                logger.warning(
                    "attempt %d/%d failed: %s", attempt, self.retry_config.max_attempts, exc
                )
                if attempt < self.retry_config.max_attempts:
                    time.sleep(self.retry_config.backoff_seconds * attempt)
        raise OrderSubmissionError(
            f"failed after {self.retry_config.max_attempts} attempts"
        ) from last_exc

    def submit_market_order(
        self, symbol: str, qty: int, side: OrderSide, client_order_id: str
    ) -> Any:
        request = MarketOrderRequest(
            symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id,
        )
        order = cast(Any, self._with_retry(self._client.submit_order, order_data=request))
        logger.info("submitted market order: %s %s %s -> order_id=%s", side, qty, symbol, order.id)
        return order

    def submit_limit_order(
        self, symbol: str, qty: int, side: OrderSide, limit_price: float, client_order_id: str
    ) -> Any:
        request = LimitOrderRequest(
            symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY,
            limit_price=limit_price, client_order_id=client_order_id,
        )
        order = cast(Any, self._with_retry(self._client.submit_order, order_data=request))
        logger.info(
            "submitted limit order: %s %s %s @ %s -> order_id=%s",
            side, qty, symbol, limit_price, order.id,
        )
        return order

    def cancel_order(self, order_id: str) -> None:
        self._with_retry(self._client.cancel_order_by_id, order_id)
        logger.info("cancelled order %s", order_id)

    def get_positions(self) -> list[Any]:
        return cast(list[Any], self._with_retry(self._client.get_all_positions))

    def get_open_orders(self) -> list[Any]:
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        return cast(list[Any], self._with_retry(self._client.get_orders, filter=request))

    def get_account_equity(self) -> float:
        account = cast(Any, self._with_retry(self._client.get_account))
        return float(account.equity)

    def is_market_open(self) -> bool:
        clock = cast(Any, self._with_retry(self._client.get_clock))
        return bool(clock.is_open)

    def close_all_positions(self) -> None:
        """Kill-switch flatten: cancel all open orders and close every position."""
        self._with_retry(self._client.cancel_orders)
        self._with_retry(self._client.close_all_positions, cancel_orders=True)
        logger.warning("kill switch: cancelled all orders and closed all positions")
