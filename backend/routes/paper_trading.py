from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.core.container import Container
from backend.routes import get_container

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/paper-trading")


class PlaceOrderRequest(BaseModel):
    symbol: str
    side: str
    quantity: float
    price: float | None = None


@router.get("/portfolio")
async def get_portfolio(container: Container = Depends(get_container)):
    return await container.paper_trading_service.get_portfolio()


@router.get("/positions")
async def get_positions(container: Container = Depends(get_container)):
    return await container.paper_trading_service.get_positions()


@router.get("/trades")
async def get_trades(
    limit: int = 50,
    container: Container = Depends(get_container),
):
    return await container.paper_trading_service.get_trade_history(limit)


@router.post("/order")
async def place_order(
    request: PlaceOrderRequest,
    container: Container = Depends(get_container),
):
    try:
        return await container.paper_trading_service.place_order(
            symbol=request.symbol,
            side=request.side,
            quantity=request.quantity,
            price=request.price,
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))


@router.post("/reset")
async def reset(container: Container = Depends(get_container)):
    return await container.paper_trading_service.reset()
