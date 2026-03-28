from datetime import datetime
from typing import Any, Optional

from backend.repositories.base import BaseRepository


class PriceRepository(BaseRepository):
    def __init__(self, db: Any) -> None:
        super().__init__(db, "price_history")

    async def get_latest(self, symbol: str) -> Optional[dict]:
        return await self.find_one(
            {"symbol": symbol.upper()},
            sort=[("timestamp", -1)],
        )

    async def get_history(self, symbol: str, since: datetime) -> list[dict]:
        return await self.find_many(
            {
                "symbol": symbol.upper(),
                "timestamp": {"$gte": since.isoformat()},
            },
            limit=10_000,
            sort=[("timestamp", 1)],
        )

    async def save(self, price_data: dict) -> Any:
        return await self.insert_one(price_data)
