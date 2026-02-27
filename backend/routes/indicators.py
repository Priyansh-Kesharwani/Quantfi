from datetime import datetime, timedelta, timezone
import logging

from fastapi import APIRouter, HTTPException, Depends

from backend.models import IndicatorData
from backend.routes import get_container
from backend.container import Container

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/indicators/{symbol}")
async def get_indicators(
    symbol: str,
    container: Container = Depends(get_container),
):
    db = container.db
    cfg = container.config
    cached = await db.indicators.find_one(
        {"symbol": symbol.upper()},
        {"_id": 0},
        sort=[("timestamp", -1)],
    )
    if cached and isinstance(cached.get("timestamp"), str):
        cached_time = datetime.fromisoformat(cached["timestamp"])
        if (
            datetime.now(timezone.utc)
            - cached_time.replace(tzinfo=timezone.utc)
            < timedelta(hours=cfg.cache_indicators_hours)
        ):
            return cached

    facade = container.scoring_facade
    indicators = facade.get_indicators_for_symbol(
        symbol.upper(), period=cfg.history_period
    )
    if indicators is None:
        raise HTTPException(
            status_code=404, detail="Unable to calculate indicators"
        )
    indicator_obj = IndicatorData(
        symbol=symbol.upper(),
        timestamp=datetime.now(timezone.utc),
        **indicators,
    )
    doc = indicator_obj.model_dump()
    doc["timestamp"] = doc["timestamp"].isoformat()
    await db.indicators.insert_one(doc)
    return indicator_obj
