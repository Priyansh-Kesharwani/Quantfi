import asyncio
from datetime import datetime, timedelta, timezone
import logging

from fastapi import APIRouter, Depends

from backend.routes import get_container
from backend.container import Container

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/news")
async def get_news(
    limit: int = 20,
    container: Container = Depends(get_container),
):
    db = container.db
    cfg = container.config
    recent_news = await db.news_events.find(
        {}, {"_id": 0}
    ).sort("published_at", -1).limit(limit).to_list(limit)
    if recent_news:
        latest = recent_news[0]
        if isinstance(latest.get("published_at"), str):
            latest_time = datetime.fromisoformat(latest["published_at"])
            if (
                datetime.now(timezone.utc)
                - latest_time.replace(tzinfo=timezone.utc)
                < timedelta(hours=cfg.cache_news_hours)
            ):
                return {"news": recent_news}
    asyncio.create_task(container.news_service.fetch_and_classify_news(db))
    return {"news": recent_news}


@router.get("/news/asset/{symbol}")
async def get_news_for_asset(
    symbol: str,
    limit: int = 3,
    container: Container = Depends(get_container),
):
    db = container.db
    symbol_upper = symbol.upper()
    query = {
        "$or": [
            {"affected_assets": symbol_upper},
            {
                "title": {
                    "$regex": symbol_upper.replace(".", "\\."),
                    "$options": "i",
                }
            },
            {
                "description": {
                    "$regex": symbol_upper.replace(".", "\\."),
                    "$options": "i",
                }
            },
        ]
    }
    results = await db.news_events.find(
        query, {"_id": 0}
    ).sort("published_at", -1).limit(limit).to_list(limit)
    if not results:
        results = await db.news_events.find(
            {}, {"_id": 0}
        ).sort("published_at", -1).limit(limit).to_list(limit)
    return {"symbol": symbol_upper, "news": results}


@router.post("/news/refresh")
async def refresh_news(container: Container = Depends(get_container)):
    await container.news_service.fetch_and_classify_news(container.db)
    return {"status": "refreshed"}
