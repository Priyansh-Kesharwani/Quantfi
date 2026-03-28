from typing import Any, Optional

from backend.repositories.base import BaseRepository


class AssetRepository(BaseRepository):
    def __init__(self, db: Any) -> None:
        super().__init__(db, "assets")

    async def get_all_active(self) -> list[dict]:
        return await self.find_many({"is_active": True})

    async def get_by_symbol(self, symbol: str) -> Optional[dict]:
        return await self.find_one({"symbol": symbol.upper()})

    async def add(self, asset_data: dict) -> Any:
        return await self.insert_one(asset_data)

    async def update(self, symbol: str, update_data: dict) -> Any:
        return await self.update_one(
            {"symbol": symbol.upper()},
            {"$set": update_data},
        )

    async def deactivate(self, symbol: str) -> Any:
        return await self.update_one(
            {"symbol": symbol.upper()},
            {"$set": {"is_active": False}},
        )

    async def reactivate(self, symbol: str, asset_data: dict) -> Any:
        return await self.update_one(
            {"symbol": symbol.upper()},
            {"$set": {**asset_data, "is_active": True}},
            upsert=True,
        )
