from datetime import datetime, timedelta, timezone
import logging

from fastapi import APIRouter, HTTPException, Depends

from backend.models import DCAScore
from backend.routes import get_container
from backend.container import Container

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/scores/{symbol}")
async def get_dca_score(
    symbol: str,
    container: Container = Depends(get_container),
):
    db = container.db
    cfg = container.config
    cached = await db.scores.find_one(
        {"symbol": symbol.upper()},
        {"_id": 0},
        sort=[("timestamp", -1)],
    )
    if cached and isinstance(cached.get("timestamp"), str):
        cached_time = datetime.fromisoformat(cached["timestamp"])
        if (
            datetime.now(timezone.utc)
            - cached_time.replace(tzinfo=timezone.utc)
            < timedelta(hours=cfg.cache_scores_hours)
        ):
            if "breakdown" in cached and isinstance(
                cached["breakdown"], dict
            ):
                return cached

    facade = container.scoring_facade
    result = facade.get_score_for_symbol(symbol.upper())
    if result is None:
        raise HTTPException(
            status_code=404, detail="Unable to calculate score"
        )
    composite_score, breakdown, top_factors = result
    zone = container.scoring_engine.get_zone(composite_score)
    explanation = await container.llm_service.generate_score_explanation(
        symbol.upper(),
        composite_score,
        breakdown.model_dump(),
        top_factors,
    )
    score_obj = DCAScore(
        symbol=symbol.upper(),
        timestamp=datetime.now(timezone.utc),
        composite_score=composite_score,
        zone=zone,
        breakdown=breakdown,
        explanation=explanation,
        top_factors=top_factors,
    )
    doc = score_obj.model_dump()
    doc["timestamp"] = doc["timestamp"].isoformat()
    doc["breakdown"] = breakdown.model_dump()
    await db.scores.insert_one(doc)
    return score_obj
