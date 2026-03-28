import asyncio

from fastapi import APIRouter, Depends

from backend.core.container import Container
from backend.routes import get_container

router = APIRouter()


@router.get("/news")
async def get_news(
    limit: int = 20,
    container: Container = Depends(get_container),
):
    cfg = container.config
    fresh = await container.news_repo.get_fresh(cfg.cache_news_hours)
    if fresh:
        return {"news": fresh[:limit]}

    recent = await container.news_repo.get_recent(limit)
    asyncio.create_task(container.news_service.fetch_and_classify_news(container.db))
    return {"news": recent}


@router.get("/news/asset/{symbol}")
async def get_news_for_asset(
    symbol: str,
    limit: int = 3,
    container: Container = Depends(get_container),
):
    results = await container.news_repo.get_for_asset(symbol, limit)
    return {"symbol": symbol.upper(), "news": results}


@router.post("/news/refresh")
async def refresh_news(container: Container = Depends(get_container)):
    await container.news_service.fetch_and_classify_news(container.db)
    return {"status": "refreshed"}
