from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Depends

from backend.models import DCAScore
from backend.core.container import Container
from backend.routes import get_container

router = APIRouter()


@router.get("/scores/{symbol}")
async def get_dca_score(
    symbol: str,
    container: Container = Depends(get_container),
):
    sym = symbol.upper()
    cfg = container.config

    cached = await container.score_repo.get_fresh(sym, cfg.cache_scores_hours)
    if cached:
        return cached

    facade = container.scoring_facade
    result = facade.get_score_for_symbol(sym)
    if result is None:
        raise HTTPException(status_code=404, detail="Unable to calculate score")

    composite_score, breakdown, top_factors = result
    zone = container.scoring_engine.get_zone(composite_score)
    explanation = await container.llm_service.generate_score_explanation(
        sym, composite_score, breakdown.model_dump(), top_factors,
    )
    score_obj = DCAScore(
        symbol=sym,
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
    await container.score_repo.save(doc)
    return score_obj
