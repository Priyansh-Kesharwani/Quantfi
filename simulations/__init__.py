"""
Phase B+3 — Hawkes Process Simulation Module.

Phase 3 additions: synthetic LOB snapshots and trade tick generation.
"""

from simulations.hawkes_simulator import (
    simulate_hawkes_events,
    ground_truth_intensity,
    branching_ratio,
    expected_event_rate,
    intensity_rmse,
    relative_rmse,
    simulate_regime,
    run_all_regimes,
    validate_estimation,
    generate_synthetic_lob,
    generate_synthetic_trades,
)

__all__ = [
    "simulate_hawkes_events",
    "ground_truth_intensity",
    "branching_ratio",
    "expected_event_rate",
    "intensity_rmse",
    "relative_rmse",
    "simulate_regime",
    "run_all_regimes",
    "validate_estimation",
    "generate_synthetic_lob",
    "generate_synthetic_trades",
]

__version__ = "0.2.0"
