from fastapi import APIRouter, HTTPException, Depends

from backend.core.container import Container
from backend.routes import get_container

router = APIRouter()


@router.get("/prices/{symbol}")
async def get_latest_price(
    symbol: str,
    container: Container = Depends(get_container),
):
    result = await container.price_service.get_latest_price(symbol)
    if not result:
        raise HTTPException(status_code=404, detail="Unable to fetch price data")
    return result


@router.get("/prices/{symbol}/history")
async def get_price_history(
    symbol: str,
    period: str = "1y",
    container: Container = Depends(get_container),
):
    result = await container.price_service.get_price_history(symbol, period)
    if not result:
        raise HTTPException(status_code=404, detail="No historical data available")
    return result
