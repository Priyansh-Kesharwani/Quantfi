"""Thin controller for crypto trading bot routes."""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from backend.core.container import Container
from backend.routes import get_container

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/crypto")


class CryptoBacktestRequest(BaseModel):
    symbol: str = "BTC/USDT:USDT"
    timeframe: str = "1h"
    initial_capital: float = 10_000.0
    strategy_mode: str = "adaptive"
    leverage: float = 3.0
    cost_preset: str = "BINANCE_FUTURES_TAKER"
    entry_threshold: float = 30.0
    exit_threshold: float = 15.0
    max_holding_bars: int = 336
    atr_trail_mult: float = 4.0
    kelly_fraction: float = 0.25
    max_risk_per_trade: float = 0.15
    score_exit_patience: int = 3
    n_bars: int = 2000
    grid_levels: int = 20
    grid_spacing: str = "geometric"
    grid_order_size: float = 100.0
    atr_multiplier: float = 3.0
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class CryptoBacktestResponse(BaseModel):
    sharpe: float
    sortino: float
    cagr: float
    max_drawdown: float
    calmar: float
    win_rate: float
    profit_factor: float
    avg_trade_pnl: float
    total_fees: float
    total_funding: float
    n_trades: int
    final_equity: float
    total_return_pct: float
    equity_curve: List[Dict[str, Any]]
    buy_hold_curve: List[Dict[str, Any]]
    price_curve: List[Dict[str, Any]]
    trades: List[Dict[str, Any]] = []
    regimes_sampled: List[str]
    regime_counts: Dict[str, int]
    baselines: Dict[str, Dict[str, float]]
    score_reachability: Dict[str, Any]
    data_source: str = "live"
    start_date: str = ""
    end_date: str = ""
    n_bars_actual: int = 0


@router.get("/markets")
async def get_markets(
    exchange: str = "binance",
    container: Container = Depends(get_container),
):
    return await container.crypto_service.get_markets(exchange)


@router.post("/backtest", response_model=CryptoBacktestResponse)
async def run_crypto_backtest(
    request: CryptoBacktestRequest,
    container: Container = Depends(get_container),
):
    try:
        return await container.crypto_service.run_backtest(request)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=502, detail=str(re))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Crypto backtest error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config/defaults")
async def get_defaults(container: Container = Depends(get_container)):
    return container.crypto_service.get_defaults()


@router.get("/strategies")
async def get_strategies(container: Container = Depends(get_container)):
    return container.crypto_service.get_strategies()
