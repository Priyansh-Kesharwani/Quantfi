from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Depends

from backend.models import IndicatorData
from backend.core.container import Container
from backend.routes import get_container

router = APIRouter()


@router.get("/indicators/{symbol}")
async def get_indicators(
    symbol: str,
    container: Container = Depends(get_container),
):
    sym = symbol.upper()
    cfg = container.config
    cached = await container.indicator_repo.get_fresh(sym, cfg.cache_indicators_hours)
    if cached:
        return cached

    facade = container.scoring_facade
    indicators = facade.get_indicators_for_symbol(sym, period=cfg.history_period)
    if indicators is None:
        raise HTTPException(status_code=404, detail="Unable to calculate indicators")

    indicator_obj = IndicatorData(
        symbol=sym,
        timestamp=datetime.now(timezone.utc),
        **indicators,
    )
    doc = indicator_obj.model_dump()
    doc["timestamp"] = doc["timestamp"].isoformat()
    await container.indicator_repo.save(doc)
    return indicator_obj
