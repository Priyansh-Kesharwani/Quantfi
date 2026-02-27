"""IExecutor protocol and shared types for order execution."""

from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable

from crypto.models import CryptoTradeRecord, FillResult, FuturesPosition, OrderIntent


@runtime_checkable
class IExecutor(Protocol):
    """Protocol for order execution backends."""

    async def place_order(self, intent: OrderIntent) -> FillResult:
        ...

    async def cancel_order(self, order_id: str) -> bool:
        ...

    async def get_position(self, symbol: str) -> Optional[FuturesPosition]:
        ...

    async def get_balance(self) -> float:
        ...

    async def close_all(self, symbol: str) -> List[CryptoTradeRecord]:
        ...
