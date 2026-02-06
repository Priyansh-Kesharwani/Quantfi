from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional
from datetime import datetime
import uuid

class Asset(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    name: str
    asset_type: str  # 'metal', 'equity', 'indian_equity'
    exchange: Optional[str] = None  # 'NSE', 'BSE', 'NASDAQ', 'NYSE', etc.
    currency: str = 'USD'  # 'USD', 'INR'
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

class ScoreBreakdown(BaseModel):
    technical_momentum: float
    volatility_opportunity: float
    statistical_deviation: float
    macro_fx: float

class DCAScore(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    timestamp: datetime
    composite_score: float  # 0-100
    zone: str  # 'unfavorable', 'neutral', 'favorable', 'strong_buy'
    breakdown: ScoreBreakdown
    explanation: str
    top_factors: List[str]

class NewsEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    source: str
    url: str
    published_at: datetime
    event_type: str  # 'rate_change', 'sanction', 'trade_restriction', 'mining_regulation', 'war', 'election'
    affected_assets: List[str]
    impact_scores: Dict[str, float]  # {symbol: confidence_score}
    summary: Optional[str] = None

class BacktestConfig(BaseModel):
    symbol: str
    start_date: datetime
    end_date: datetime
    dca_amount: float
    dca_cadence: str  # 'weekly' or 'monthly'
    buy_dip_threshold: Optional[float] = 60  # Score threshold for extra buying

class BacktestResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    config: BacktestConfig
    total_invested: float
    total_units: float
    final_value_usd: float
    final_value_inr: float
    total_return_pct: float
    annualized_return_pct: float
    num_regular_dca: int
    num_dip_buys: int
    created_at: datetime = Field(default_factory=datetime.utcnow)

class UserSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    default_dca_cadence: str = 'monthly'
    default_dca_amount: float = 5000
    dip_alert_threshold: float = 70
    score_weights: Dict[str, float] = {
        'technical_momentum': 0.4,
        'volatility_opportunity': 0.2,
        'statistical_deviation': 0.2,
        'macro_fx': 0.2
    }


# PHASE1: indicator hook - Phase 1 Indicator Models
# These models support the Phase 1 Market State Indicator Engine

class Phase1IndicatorMeta(BaseModel):
    """Metadata for explainability and audit trail"""
    model_config = ConfigDict(extra="ignore")
    
    window_used: int
    n_obs: int
    method: str
    seed: Optional[int] = None
    notes: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Phase1NormalizedComponent(BaseModel):
    """A normalized component score with metadata"""
    model_config = ConfigDict(extra="ignore")
    
    raw_value: float
    normalized_value: float  # In [0, 1]
    percentile: float
    ecdf_sample_size: int
    meta: Phase1IndicatorMeta


class Phase1CompositeResult(BaseModel):
    """Full Phase 1 composite score with intermediates for audit"""
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Normalized components (all in [0, 1])
    T_t: float  # Trend
    U_t: float  # Undervaluation (VWAP Z)
    V_t: float  # Volatility regime
    L_t: float  # Liquidity
    C_t: float  # Systemic coupling
    H_t: float  # Hurst exponent
    R_t: float  # HMM regime probability
    
    # Intermediate calculations
    g_pers_H: float  # Persistence modifier
    Gate_t: float    # C_t * L_t * R_t_thresholded
    Opp_t: float     # Committee aggregation
    RawFavor_t: float  # Opp_t * Gate_t
    
    # Final score
    composite_score: float  # In [0, 100], anchored at 50
    
    # Metadata
    config_used: Dict[str, Optional[float]] = {}  # Symbolic placeholders
    notes: str = ""
