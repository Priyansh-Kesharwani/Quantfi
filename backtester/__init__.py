"""
Backtester Module - Phase 2

Diagnostic-only backtesting framework for validating the composite score.

This is NOT for predictive PnL optimization. The goal is to validate that
high composite scores historically correspond to favorable DCA conditions.

Evaluations:
- Score deciles vs forward returns
- Score vs forward volatility
- DCA cost comparison (high-score vs uniform schedule)
- Drawdown avoidance statistics
- Crisis regime overlays

Author: Phase 2 Implementation
Date: 2026-02-07
"""

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

__all__ = [
    # Main classes
    'DiagnosticBacktester',
    'BacktestConfig',
    'BacktestResult',
    'DecileAnalysis',
    'DCAComparison',
    'CrisisOverlay',
    # Evaluation functions
    'score_vs_forward_returns',
    'score_vs_forward_volatility', 
    'dca_cost_comparison',
    'drawdown_analysis',
    'crisis_regime_analysis'
]
