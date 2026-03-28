from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from backend.repositories.base import BaseRepository


class NewsRepository(BaseRepository):
    def __init__(self, db: Any) -> None:
        super().__init__(db, "news_events")

    async def get_recent(self, limit: int = 20) -> list[dict]:
        return await self.find_many(
            {},
            limit=limit,
            sort=[("published_at", -1)],
        )

    async def get_for_asset(self, symbol: str, limit: int = 3) -> list[dict]:
        symbol_upper = symbol.upper()
        escaped = symbol_upper.replace(".", "\\.")
        query = {
            "$or": [
                {"affected_assets": symbol_upper},
                {"title": {"$regex": escaped, "$options": "i"}},
                {"description": {"$regex": escaped, "$options": "i"}},
            ]
        }
        results = await self.find_many(
            query,
            limit=limit,
            sort=[("published_at", -1)],
        )
        if not results:
            results = await self.get_recent(limit=limit)
        return results

    async def get_fresh(self, max_age_hours: int) -> Optional[list[dict]]:
        recent = await self.get_recent(limit=1)
        if not recent:
            return None
        latest = recent[0]
        ts = latest.get("published_at")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        if ts and (
            datetime.now(timezone.utc) - ts.replace(tzinfo=timezone.utc)
            < timedelta(hours=max_age_hours)
        ):
            return await self.get_recent()
        return None

    async def save_many(self, events: list[dict]) -> Any:
        if not events:
            return None
        return await self._collection.insert_many(events)
