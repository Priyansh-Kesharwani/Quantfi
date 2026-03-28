from typing import Any, Optional

from backend.repositories.base import BaseRepository


class SettingsRepository(BaseRepository):
    def __init__(self, db: Any) -> None:
        super().__init__(db, "user_settings")

    async def get(self) -> Optional[dict]:
        return await self.find_one({})

    async def save(self, settings_data: dict) -> Any:
        return await self.update_one(
            {},
            {"$set": settings_data},
            upsert=True,
        )
