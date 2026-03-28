from .diagnostics import (
    DiagnosticBacktester,
    BacktestConfig,
    BacktestResult,
    DecileAnalysis,
    DCAComparison,
    CrisisOverlay
)

from .evaluations import (
    score_vs_forward_returns,
    score_vs_forward_volatility,
    dca_cost_comparison,
    drawdown_analysis,
    crisis_regime_analysis
)

from .purged_validation import (
    PurgedKFold,
    WalkForwardCV,
    BlockBootstrap,
    FoldResult,
    walk_forward_validate,
    wfo_rolling_2y1y_splits,
    run_wfo_rolling_harness,
    WFOFoldResult,
)

from .signal_sweep import (
    SweepConfig,
    sweep_dca_thresholds,
    sweep_cadence_and_threshold,
    compute_sweep_heatmap,
    rank_sweep_results
)

from .dca_portfolio_sim import (
    DCAPortfolioConfig,
    DCAPortfolioSimulator,
    PortfolioSimResult,
    simulate_multi_asset_dca,
    generate_dca_report
)

from .portfolio_simulator import (
    PortfolioSimulator,
    SimConfig,
    ExitParams,
    prepare_multi_asset_data,
    COST_PRESETS,
)

from .allocation_engine import (
    AllocationConfig,
    AllocationEngine,
    RegimeHysteresis,
    RISK_ON,
    RISK_OFF,
    compute_target_weights,
)

__all__ = [
    'DiagnosticBacktester',
    'BacktestConfig',
    'BacktestResult',
    'DecileAnalysis',
    'DCAComparison',
    'CrisisOverlay',
    'score_vs_forward_returns',
    'score_vs_forward_volatility',
    'dca_cost_comparison',
    'drawdown_analysis',
    'crisis_regime_analysis',
    'PurgedKFold',
    'WalkForwardCV',
    'BlockBootstrap',
    'FoldResult',
    'walk_forward_validate',
    'wfo_rolling_2y1y_splits',
    'run_wfo_rolling_harness',
    'WFOFoldResult',
    'SweepConfig',
    'sweep_dca_thresholds',
    'sweep_cadence_and_threshold',
    'compute_sweep_heatmap',
    'rank_sweep_results',
    'DCAPortfolioConfig',
    'DCAPortfolioSimulator',
    'PortfolioSimResult',
    'simulate_multi_asset_dca',
    'generate_dca_report',
    'PortfolioSimulator',
    'SimConfig',
    'ExitParams',
    'prepare_multi_asset_data',
    'COST_PRESETS',
    'AllocationConfig',
    'AllocationEngine',
    'RegimeHysteresis',
    'RISK_ON',
    'RISK_OFF',
    'compute_target_weights',
]
