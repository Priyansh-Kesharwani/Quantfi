from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from backend.repositories.base import BaseRepository


class ScoreRepository(BaseRepository):
    def __init__(self, db: Any) -> None:
        super().__init__(db, "scores")

    async def get_latest(self, symbol: str) -> Optional[dict]:
        return await self.find_one(
            {"symbol": symbol.upper()},
            sort=[("timestamp", -1)],
        )

    async def get_fresh(
        self, symbol: str, max_age_hours: int
    ) -> Optional[dict]:
        cached = await self.get_latest(symbol)
        if not cached:
            return None
        ts = cached.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        if ts and (
            datetime.now(timezone.utc) - ts.replace(tzinfo=timezone.utc)
            < timedelta(hours=max_age_hours)
        ):
            if isinstance(cached.get("breakdown"), dict):
                return cached
        return None

    async def save(self, score_data: dict) -> Any:
        return await self.insert_one(score_data)
