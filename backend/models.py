from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional
from datetime import datetime
import uuid


def _cfg():
    from backend.app_config import get_backend_config
    return get_backend_config()


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
    composite_score: float
    zone: str
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
    event_type: str
    affected_assets: List[str]
    impact_scores: Dict[str, float]
    summary: Optional[str] = None

class BacktestConfig(BaseModel):
    symbol: str
    start_date: datetime
    end_date: datetime
    dca_amount: float
    dca_cadence: str
    buy_dip_threshold: Optional[float] = None

    def model_post_init(self, __context) -> None:
        if self.buy_dip_threshold is None:
            self.buy_dip_threshold = _cfg().backtest_buy_dip_threshold_default


class EquityPoint(BaseModel):
    date: str
    portfolio_value: float
    total_invested: float
    price: float
    score: Optional[float] = None
    is_dip_buy: bool = False


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
    max_drawdown_pct: float = 0.0
    avg_cost_basis: float = 0.0
    data_points: int = 0
    data_source: str = "yfinance"
    data_start: Optional[str] = None
    data_end: Optional[str] = None
    equity_curve: List[EquityPoint] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class UserSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    default_dca_cadence: str = 'monthly'
    default_dca_amount: Optional[float] = None
    dip_alert_threshold: Optional[float] = None
    score_weights: Optional[Dict[str, float]] = None

    def model_post_init(self, __context) -> None:
        cfg = _cfg()
        if self.default_dca_amount is None:
            self.default_dca_amount = cfg.user_default_dca_amount
        if self.dip_alert_threshold is None:
            self.dip_alert_threshold = cfg.user_default_dip_alert_threshold
        if self.score_weights is None:
            self.score_weights = dict(cfg.default_score_weights)


class SimulationExitConfig(BaseModel):
    atr_init_mult: float = 2.0
    atr_trail_mult: float = 2.5
    min_stop_pct: float = 4.0
    score_rel_mult: float = 0.4
    score_abs_floor: float = 35.0
    max_holding_days: int = 30

class SimulationRequest(BaseModel):
    symbols: List[str]
    start_date: str
    end_date: str
    initial_capital: float = 100000.0
    entry_score_threshold: float = 70.0
    exit_config: Optional[SimulationExitConfig] = None
    max_positions: int = 10
    use_score_weighting: bool = True
    min_position_notional: float = 3000.0
    slippage_bps: float = 5.0
    run_benchmarks: bool = True
    template: Optional[str] = None
    cost_free: bool = False
    export_trades: bool = False
    run_id: Optional[str] = None
    min_invested_fraction: float = 0.0
    scoring_mode: str = "adaptive"
    simulation_mode: str = "tactical"
    risk_on_equity_pct: float = 0.95
    risk_off_equity_pct: float = 0.60
    theta_tilt: float = 0.0
    rebalance_freq_days: int = 42
    min_rebalance_delta: float = 0.03
    jump_penalty: float = 25.0
    drawdown_circuit_threshold: float = -0.15
    cash_return_annual: float = 0.02


class Phase1IndicatorMeta(BaseModel):
    model_config = ConfigDict(extra="ignore")

    window_used: int
    n_obs: int
    method: str
    seed: Optional[int] = None
    notes: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Phase1NormalizedComponent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    raw_value: float
    normalized_value: float
    percentile: float
    ecdf_sample_size: int
    meta: Phase1IndicatorMeta


class Phase1CompositeResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    T_t: float
    U_t: float
    V_t: float
    L_t: float
    C_t: float
    H_t: float
    R_t: float

    g_pers_H: float
    Gate_t: float
    Opp_t: float
    RawFavor_t: float

    composite_score: float

    config_used: Dict[str, Optional[float]] = {}
    notes: str = ""
