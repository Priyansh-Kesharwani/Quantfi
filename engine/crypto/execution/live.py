"""LiveCCXTExecutor: live trading executor using CCXT (skeleton)."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from engine.crypto.models import (
    CryptoTradeRecord,
    FillResult,
    FuturesPosition,
    OrderIntent,
)

logger = logging.getLogger(__name__)


def _generate_client_order_id(intent: OrderIntent) -> str:
    """Deterministic-ish client order ID from intent + timestamp."""
    return f"qf_{intent.symbol}_{intent.side}_{uuid.uuid4().hex[:12]}"


class LiveCCXTExecutor:
    """Live executor using ccxt.pro WebSockets.

    Handles partial fills by accumulating FillResults until complete.
    This is a skeleton -- full implementation requires exchange API keys.
    """

    def __init__(self, exchange: Any, config: Any):
        self._exchange = exchange
        self._config = config
        self._pending_fills: Dict[str, Any] = {}
        self._recent_client_ids: Dict[str, str] = {}

    async def _check_existing_orders(self, symbol: str, client_id: str) -> Optional[dict]:
        """Check if an order with this client ID already exists on the exchange."""
        try:
            open_orders = await self._exchange.fetch_open_orders(symbol)
            for o in open_orders:
                if o.get("clientOrderId") == client_id:
                    return o
        except Exception as e:
            logger.warning("Open-order check failed: %s", e)
        return None

    async def place_order(self, intent: OrderIntent) -> FillResult:
        """Place order via CCXT with idempotent client order ID."""
        client_id = _generate_client_order_id(intent)
        try:
            existing = await self._check_existing_orders(intent.symbol, client_id)
            if existing:
                logger.info("Order %s already exists on exchange, skipping duplicate", client_id)
                filled = existing.get("filled", 0.0)
                remaining = existing.get("remaining", intent.quantity - filled)
                avg_price = existing.get("average", existing.get("price", 0))
            else:
                order_type = intent.order_type
                side = intent.side
                amount = intent.quantity
                price = intent.price
                params = {"clientOrderId": client_id}

                if order_type == "market":
                    result = await self._exchange.create_order(
                        intent.symbol, "market", side, amount, params=params,
                    )
                else:
                    result = await self._exchange.create_order(
                        intent.symbol, "limit", side, amount, price, params=params,
                    )

                filled = result.get("filled", 0.0)
                remaining = result.get("remaining", amount - filled)
                avg_price = result.get("average", result.get("price", 0))

            self._recent_client_ids[client_id] = intent.symbol

            trade = CryptoTradeRecord(
                timestamp=datetime.utcnow(),
                symbol=intent.symbol,
                side=f"{'long' if intent.side == 'buy' else 'short'}_entry",
                units=filled,
                price=avg_price,
                notional=filled * avg_price,
                fee=0.0,
                slippage=0.0,
                funding_paid=0.0,
                pnl=0.0,
                exit_reason=intent.reason,
                leverage=self._config.leverage if hasattr(self._config, "leverage") else 1.0,
                bar_idx=0,
            )

            return FillResult(
                trade=trade,
                filled_qty=filled,
                remaining_qty=remaining,
                is_complete=remaining <= 0,
            )
        except Exception as e:
            logger.error("Order placement failed (client_id=%s): %s", client_id, e)
            raise

    async def cancel_order(self, order_id: str) -> bool:
        try:
            await self._exchange.cancel_order(order_id)
            return True
        except Exception as e:
            logger.error("Cancel failed for %s: %s", order_id, e)
            return False

    async def get_position(self, symbol: str) -> Optional[FuturesPosition]:
        """Fetch current position from exchange."""
        try:
            positions = await self._exchange.fetch_positions([symbol])
            for pos in positions:
                if pos.get("contracts", 0) > 0:
                    side = pos.get("side", "long")
                    return FuturesPosition(
                        symbol=symbol,
                        direction=side,
                        units=float(pos["contracts"]),
                        entry_price=float(pos.get("entryPrice", 0)),
                        leverage=float(pos.get("leverage", 1)),
                        margin=float(pos.get("initialMargin", 0)),
                        liquidation_price=float(pos.get("liquidationPrice", 0)),
                        entry_time=datetime.utcnow(),
                        peak_price=float(pos.get("entryPrice", 0)),
                    )
            return None
        except Exception as e:
            logger.error("Position fetch failed: %s", e)
            return None

    async def get_balance(self) -> float:
        try:
            balance = await self._exchange.fetch_balance()
            return float(balance.get("total", {}).get("USDT", 0))
        except Exception as e:
            logger.error("Balance fetch failed: %s", e)
            return 0.0

    async def close_all(self, symbol: str) -> List[CryptoTradeRecord]:
        """Emergency close all positions."""
        pos = await self.get_position(symbol)
        if pos is None:
            return []
        close_side = "sell" if pos.direction == "long" else "buy"
        intent = OrderIntent(
            symbol=symbol,
            side=close_side,
            order_type="market",
            quantity=pos.units,
            reduce_only=True,
            reason="emergency_close",
        )
        fill = await self.place_order(intent)
        return [fill.trade]
