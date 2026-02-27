from datetime import datetime
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from backend.models import BacktestResult
from backend.routes import get_container
from backend.container import Container

logger = logging.getLogger(__name__)
router = APIRouter()


class BacktestRequest(BaseModel):
    symbol: str
    start_date: str
    end_date: str
    dca_amount: float
    dca_cadence: str
    buy_dip_threshold: Optional[float] = None


@router.post("/backtest", response_model=BacktestResult)
async def run_backtest(
    request: BacktestRequest,
    container: Container = Depends(get_container),
):
    try:
        cfg = container.config
        if request.buy_dip_threshold is None:
            request.buy_dip_threshold = cfg.backtest_buy_dip_threshold_default
        result = container.backtest_service.run_backtest(
            symbol=request.symbol.upper(),
            start_date=datetime.fromisoformat(request.start_date),
            end_date=datetime.fromisoformat(request.end_date),
            dca_amount=request.dca_amount,
            dca_cadence=request.dca_cadence,
            buy_dip_threshold=request.buy_dip_threshold,
        )
        db = container.db
        try:
            doc = result.model_dump()
            doc["created_at"] = doc["created_at"].isoformat()
            doc["config"] = {
                "symbol": result.config.symbol,
                "start_date": result.config.start_date.isoformat(),
                "end_date": result.config.end_date.isoformat(),
                "dca_amount": result.config.dca_amount,
                "dca_cadence": result.config.dca_cadence,
                "buy_dip_threshold": result.config.buy_dip_threshold,
            }
            doc.pop("equity_curve", None)
            await db.backtest_results.insert_one(doc)
        except Exception as db_err:
            logger.warning("Failed to save backtest to DB: %s", db_err)
        return result
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error("Backtest error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
