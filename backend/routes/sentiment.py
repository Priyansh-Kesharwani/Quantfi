from fastapi import APIRouter, Depends

from backend.core.container import Container
from backend.routes import get_container

router = APIRouter()


@router.get("/sentiment/{symbol}")
async def get_sentiment(
    symbol: str,
    container: Container = Depends(get_container),
):
    return await container.sentiment_service.get_sentiment(symbol)


@router.post("/sentiment/{symbol}")
async def run_sentiment(
    symbol: str,
    container: Container = Depends(get_container),
):
    return await container.sentiment_service.run_analysis(symbol, container.db)
