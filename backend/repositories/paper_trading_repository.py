from typing import Any, Optional

from backend.repositories.base import BaseRepository


class PaperTradingRepository:
    """Wraps three collections: portfolios, trades, and positions."""

    def __init__(self, db: Any) -> None:
        self._portfolios = BaseRepository(db, "paper_portfolios")
        self._trades = BaseRepository(db, "paper_trades")
        self._positions = BaseRepository(db, "paper_positions")

    async def get_portfolio(self) -> Optional[dict]:
        return await self._portfolios.find_one({})

    async def save_portfolio(self, portfolio_data: dict) -> Any:
        return await self._portfolios.update_one(
            {},
            {"$set": portfolio_data},
            upsert=True,
        )

    async def add_trade(self, trade_data: dict) -> Any:
        return await self._trades.insert_one(trade_data)

    async def get_trades(self, limit: int = 50) -> list[dict]:
        return await self._trades.find_many(
            {},
            limit=limit,
            sort=[("timestamp", -1)],
        )

    async def get_positions(self) -> list[dict]:
        return await self._positions.find_many({})

    async def save_position(self, position_data: dict) -> Any:
        symbol = position_data.get("symbol", "")
        return await self._positions.update_one(
            {"symbol": symbol},
            {"$set": position_data},
            upsert=True,
        )

    async def delete_position(self, symbol: str) -> Any:
        return await self._positions.delete_one({"symbol": symbol})

    async def reset(self) -> None:
        await self._portfolios.delete_many({})
        await self._trades.delete_many({})
        await self._positions.delete_many({})
