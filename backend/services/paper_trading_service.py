from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from backend.repositories.paper_trading_repository import PaperTradingRepository


class PaperTradingService:
    DEFAULT_CAPITAL = 100000.0

    def __init__(self, paper_repo: PaperTradingRepository, price_adapter) -> None:
        self._repo = paper_repo
        self._prices = price_adapter

    async def get_portfolio(self) -> dict:
        portfolio = await self._repo.get_portfolio()
        if not portfolio:
            portfolio = {
                "cash": self.DEFAULT_CAPITAL,
                "initial_capital": self.DEFAULT_CAPITAL,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            await self._repo.save_portfolio(portfolio)
        return portfolio

    async def get_positions(self) -> list:
        return await self._repo.get_positions()

    async def get_trade_history(self, limit: int = 50) -> list:
        return await self._repo.get_trades(limit)

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
    ) -> dict:
        portfolio = await self.get_portfolio()
        if price is None:
            price_data = await self._prices.fetch_latest_price(symbol)
            price = price_data.get("price_usd", 0) if price_data else 0

        total_cost = price * quantity
        if side == "buy":
            if portfolio["cash"] < total_cost:
                raise ValueError("Insufficient cash")
            portfolio["cash"] -= total_cost
        elif side == "sell":
            portfolio["cash"] += total_cost

        trade = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "total": total_cost,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await self._repo.add_trade(trade)

        positions = await self._repo.get_positions()
        existing = next((p for p in positions if p.get("symbol") == symbol), None)

        if side == "buy":
            if existing:
                new_qty = existing["quantity"] + quantity
                avg_price = (
                    existing["avg_price"] * existing["quantity"] + price * quantity
                ) / new_qty
                existing["quantity"] = new_qty
                existing["avg_price"] = avg_price
                await self._repo.save_position(existing)
            else:
                await self._repo.save_position({
                    "symbol": symbol,
                    "quantity": quantity,
                    "avg_price": price,
                    "opened_at": datetime.now(timezone.utc).isoformat(),
                })
        elif side == "sell" and existing:
            remaining = existing["quantity"] - quantity
            if remaining <= 0:
                await self._repo.delete_position(symbol)
            else:
                existing["quantity"] = remaining
                await self._repo.save_position(existing)

        await self._repo.save_portfolio(portfolio)
        return trade

    async def reset(self):
        await self._repo.reset()
        return await self.get_portfolio()
