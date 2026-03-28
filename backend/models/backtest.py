from typing import List, Optional
from datetime import datetime
import uuid

from pydantic import BaseModel, ConfigDict, Field


def _cfg():
    from backend.core.config import get_backend_config
    return get_backend_config()


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
