import asyncio
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from backend.models import Asset
from backend.core.container import Container
from backend.routes import get_container

logger = logging.getLogger(__name__)
router = APIRouter()


class AddAssetRequest(BaseModel):
    symbol: str
    name: str
    asset_type: str
    exchange: Optional[str] = None
    currency: str = "USD"


@router.post("/assets", response_model=Asset)
async def add_asset(
    request: AddAssetRequest,
    container: Container = Depends(get_container),
):
    symbol = request.symbol.upper().strip()
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")

    repo = container.asset_repo
    existing = await repo.get_by_symbol(symbol)

    if existing:
        if existing.get("is_active", True):
            logger.info("Asset %s already exists and is active", symbol)
            if isinstance(existing.get("created_at"), str):
                existing["created_at"] = datetime.fromisoformat(existing["created_at"])
            return Asset(**existing)
        logger.info("Reactivating soft-deleted asset %s", symbol)
        await repo.reactivate(symbol, {
            "name": request.name or existing.get("name", symbol),
            "asset_type": request.asset_type or existing.get("asset_type", "equity"),
            "exchange": request.exchange or existing.get("exchange"),
            "currency": request.currency or existing.get("currency", "USD"),
        })
        reactivated = await repo.get_by_symbol(symbol)
        if isinstance(reactivated.get("created_at"), str):
            reactivated["created_at"] = datetime.fromisoformat(reactivated["created_at"])
        asyncio.create_task(
            container.asset_data_service.fetch_and_calculate_asset_data(
                container.db, symbol, request.exchange, request.currency
            )
        )
        return Asset(**reactivated)

    asset = Asset(
        symbol=symbol,
        name=request.name,
        asset_type=request.asset_type,
        exchange=request.exchange,
        currency=request.currency,
    )
    doc = asset.model_dump()
    doc["created_at"] = doc["created_at"].isoformat()
    await repo.add(doc)
    asyncio.create_task(
        container.asset_data_service.fetch_and_calculate_asset_data(
            container.db, symbol, request.exchange, request.currency
        )
    )
    logger.info("Added new asset %s to watchlist", symbol)
    return asset


@router.get("/assets", response_model=list)
async def get_assets(container: Container = Depends(get_container)):
    assets = await container.asset_repo.get_all_active()
    for asset in assets:
        if isinstance(asset.get("created_at"), str):
            asset["created_at"] = datetime.fromisoformat(asset["created_at"])
    return assets


@router.delete("/assets/{symbol}")
async def remove_asset(
    symbol: str,
    container: Container = Depends(get_container),
):
    result = await container.asset_repo.deactivate(symbol)
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Asset not found")
    return {"status": "removed"}
