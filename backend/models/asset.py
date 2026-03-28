from typing import Optional
from datetime import datetime
import uuid

from pydantic import BaseModel, ConfigDict, Field


class Asset(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    name: str
    asset_type: str
    exchange: Optional[str] = None
    currency: str = 'USD'
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PriceData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    timestamp: datetime
    price_usd: float
    price_inr: float
    usd_inr_rate: float
    volume: Optional[float] = None


class IndicatorData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    timestamp: datetime
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_50: Optional[float] = None
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    atr_14: Optional[float] = None
    atr_percentile: Optional[float] = None
    z_score_20: Optional[float] = None
    z_score_50: Optional[float] = None
    z_score_100: Optional[float] = None
    drawdown_pct: Optional[float] = None
    adx_14: Optional[float] = None
