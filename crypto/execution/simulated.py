"""SimulatedExecutor: instant-fill executor for backtesting."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from crypto.costs import execution_price, trade_cost
from crypto.engines.futures_engine import FuturesEngineConfig
from crypto.models import (
    CryptoTradeRecord,
    FillResult,
    FuturesPosition,
    OrderIntent,
)


class SimulatedExecutor:
    """Backtesting executor with instant fills and configurable costs.

    FillResult.is_complete is always True (no partial fills in simulation).
    """

    def __init__(self, config: FuturesEngineConfig, current_price_fn=None):
        self._config = config
        self._current_price_fn = current_price_fn or (lambda: 50_000.0)
        self._balance = 10_000.0
        self._position: Optional[FuturesPosition] = None

    def set_balance(self, balance: float) -> None:
        self._balance = balance

    def set_position(self, position: Optional[FuturesPosition]) -> None:
        self._position = position

    async def place_order(self, intent: OrderIntent) -> FillResult:
        price = self._current_price_fn()
        exec_price = execution_price(price, intent.side, preset=self._config.cost_preset)
        notional = exec_price * intent.quantity
        fee = trade_cost(notional, self._config.cost_preset)

        trade = CryptoTradeRecord(
            timestamp=datetime.utcnow(),
            symbol=intent.symbol,
            side="long_entry" if intent.side == "buy" else "short_entry",
            units=intent.quantity,
            price=exec_price,
            notional=notional,
            fee=fee,
            slippage=abs(exec_price - price) * intent.quantity,
            funding_paid=0.0,
            pnl=0.0,
            exit_reason=intent.reason,
            leverage=self._config.leverage,
            bar_idx=0,
        )

        return FillResult(
            trade=trade,
            filled_qty=intent.quantity,
            remaining_qty=0.0,
            is_complete=True,
        )

    async def cancel_order(self, order_id: str) -> bool:
        return True

    async def get_position(self, symbol: str) -> Optional[FuturesPosition]:
        return self._position

    async def get_balance(self) -> float:
        return self._balance

    async def close_all(self, symbol: str) -> List[CryptoTradeRecord]:
        return []
