import asyncio
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from backend.models import Asset
from backend.routes import get_container
from backend.container import Container

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
    db = container.db

    existing = await db.assets.find_one({"symbol": symbol}, {"_id": 0})
    if existing:
        if existing.get("is_active", True):
            logger.info("Asset %s already exists and is active", symbol)
            return Asset(**existing)
        logger.info("Reactivating soft-deleted asset %s", symbol)
        await db.assets.update_one(
            {"symbol": symbol},
            {
                "$set": {
                    "is_active": True,
                    "name": request.name or existing.get("name", symbol),
                    "asset_type": request.asset_type
                    or existing.get("asset_type", "equity"),
                    "exchange": request.exchange or existing.get("exchange"),
                    "currency": request.currency
                    or existing.get("currency", "USD"),
                }
            },
        )
        reactivated = await db.assets.find_one({"symbol": symbol}, {"_id": 0})
        if isinstance(reactivated.get("created_at"), str):
            reactivated["created_at"] = datetime.fromisoformat(
                reactivated["created_at"]
            )
        asyncio.create_task(
            container.asset_data_service.fetch_and_calculate_asset_data(
                db, symbol, request.exchange, request.currency
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
    await db.assets.insert_one(doc)
    asyncio.create_task(
        container.asset_data_service.fetch_and_calculate_asset_data(
            db, symbol, request.exchange, request.currency
        )
    )
    logger.info("Added new asset %s to watchlist", symbol)
    return asset


@router.get("/assets", response_model=list)
async def get_assets(container: Container = Depends(get_container)):
    db = container.db
    assets = await db.assets.find(
        {"is_active": True}, {"_id": 0}
    ).to_list(100)
    for asset in assets:
        if isinstance(asset.get("created_at"), str):
            asset["created_at"] = datetime.fromisoformat(asset["created_at"])
    return assets


@router.delete("/assets/{symbol}")
async def remove_asset(
    symbol: str,
    container: Container = Depends(get_container),
):
    db = container.db
    result = await db.assets.update_one(
        {"symbol": symbol.upper()},
        {"$set": {"is_active": False}},
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Asset not found")
    return {"status": "removed"}
