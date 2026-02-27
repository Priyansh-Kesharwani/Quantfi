"""
Phase B+3 — Robust Validation, Simulation & Tuning Framework.

Modules
-------
metrics          : Score and signal evaluation functions.
walkforward      : Walk-forward cross-validation engine.
kfold            : Purged K-fold CV with embargo.
data_integrity   : DataFrame validation and sanity checks.
plots            : Visualization utilities.
report_generator : HTML report builder.
execution_model  : Slippage, market-impact & transaction-cost model (Phase 3).
tuning           : Nested CV hyperparameter tuning engine (Phase 3).
phase3_runner    : End-to-end orchestrator (Phase 3).
"""

from validation.metrics import (
    information_coefficient,
    hit_rate,
    sortino_ratio,
    max_drawdown,
    cagr,
    forward_returns,
    evaluate_signals,
    compute_score_metrics,
    compute_all_metrics,
)

from validation.walkforward import walkforward_cv, WalkForwardResult, WalkForwardFoldResult, WalkForwardConfig
from validation.kfold import purged_kfold, KFoldResult, KFoldFoldResult, PurgedKFoldConfig
from validation.data_integrity import (
    validate_dataframe, clean_dataframe, canonicalize,
    fetch_and_validate, load_phaseB_config,
)
from validation.report_generator import generate_report
from validation.execution_model import (
    ExecutionConfig, market_impact, compute_fill_price,
    apply_execution_costs, slippage_sensitivity_matrix,
    simulate_latency,
)
from validation.tuning import (
    TuningConfig, TuningResult, TuningTrialResult,
    run_tuning, parameter_sensitivity, ablation_study,
)
from validation.validator import CPCVConfig, CPCVSplit, generate_cpcv_splits, write_splits_metadata
from validation.objective import compute_gt_score, compute_dsr, equity_curve_from_result
from validation.orchestrator import run_orchestrator, OrchestratorResult

__all__ = [
    # Metrics
    "information_coefficient",
    "hit_rate",
    "sortino_ratio",
    "max_drawdown",
    "cagr",
    "forward_returns",
    "evaluate_signals",
    "compute_score_metrics",
    "compute_all_metrics",
    # Walk-forward
    "walkforward_cv",
    "WalkForwardResult",
    "WalkForwardFoldResult",
    "WalkForwardConfig",
    # K-fold
    "purged_kfold",
    "KFoldResult",
    "KFoldFoldResult",
    "PurgedKFoldConfig",
    # Data integrity
    "validate_dataframe",
    "clean_dataframe",
    "canonicalize",
    "fetch_and_validate",
    "load_phaseB_config",
    # Reporting
    "generate_report",
    # Execution model (Phase 3)
    "ExecutionConfig",
    "market_impact",
    "compute_fill_price",
    "apply_execution_costs",
    "slippage_sensitivity_matrix",
    "simulate_latency",
    # Tuning (Phase 3)
    "TuningConfig",
    "TuningResult",
    "TuningTrialResult",
    "run_tuning",
    "parameter_sensitivity",
    "ablation_study",
    # CPCV / GT-Score / MFBO
    "CPCVConfig",
    "CPCVSplit",
    "generate_cpcv_splits",
    "write_splits_metadata",
    "compute_gt_score",
    "compute_dsr",
    "equity_curve_from_result",
    "run_orchestrator",
    "OrchestratorResult",
]

__version__ = "0.2.0"
