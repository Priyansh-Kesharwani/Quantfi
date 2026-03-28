from typing import List, Optional

from pydantic import BaseModel


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
