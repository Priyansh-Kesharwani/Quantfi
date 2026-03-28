from typing import List, Dict, Optional
from datetime import datetime
import uuid

from pydantic import BaseModel, ConfigDict, Field


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
