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
    walk_forward_validate
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
]
