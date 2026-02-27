from pathlib import Path
from datetime import datetime, timedelta, timezone
import logging
import importlib.util
import sys

from fastapi import APIRouter, HTTPException, Depends

from backend.routes import get_container
from backend.container import Container

logger = logging.getLogger(__name__)
router = APIRouter()

_sentiment_mod = None


def _get_sentiment_mod():
    global _sentiment_mod
    if _sentiment_mod is None:
        root = Path(__file__).resolve().parent.parent.parent
        sa_path = str(root / "indicators" / "sentiment_agent.py")
        spec = importlib.util.spec_from_file_location(
            "sentiment_agent", sa_path
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["sentiment_agent"] = mod
        spec.loader.exec_module(mod)
        _sentiment_mod = mod
    return _sentiment_mod


@router.get("/sentiment/{symbol}")
async def get_sentiment(
    symbol: str,
    container: Container = Depends(get_container),
):
    db = container.db
    cfg = container.config
    sym = symbol.upper()
    cached = await db.sentiment.find_one(
        {"symbol": sym}, {"_id": 0}, sort=[("timestamp", -1)]
    )
    if cached:
        ts = cached.get("timestamp", "")
        if isinstance(ts, str):
            try:
                age = datetime.now(timezone.utc) - datetime.fromisoformat(
                    ts
                ).replace(tzinfo=timezone.utc)
                if age < timedelta(hours=cfg.cache_scores_hours):
                    return cached
            except Exception:
                pass
        return cached
    return {
        "symbol": sym,
        "status": "not_analyzed",
        "G_t": 1.0,
        "confidence": 0,
    }


@router.post("/sentiment/{symbol}")
async def run_sentiment(
    symbol: str,
    container: Container = Depends(get_container),
):
    db = container.db
    sym = symbol.upper()
    try:
        sa = _get_sentiment_mod()
        result = await sa.full_sentiment_analysis(sym)
        doc = result.model_dump()
        doc["symbol"] = sym
        doc["timestamp"] = datetime.now(timezone.utc).isoformat()
        doc["top_factors"] = [f.model_dump() for f in result.top_factors]
        doc["citations"] = [c.model_dump() for c in result.citations]
        store_doc = {k: v for k, v in doc.items()}
        await db.sentiment.insert_one(store_doc)
        doc.pop("_id", None)
        return doc
    except Exception as e:
        logger.error("Sentiment analysis failed for %s: %s", sym, e)
        return {
            "symbol": sym,
            "G_t": 1.0,
            "raw_sentiment": 0.0,
            "confidence": 0.0,
            "reasoning": f"Analysis unavailable: {str(e)}",
            "top_factors": [],
            "citations": [],
        }

