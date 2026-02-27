"""Backward-compatible re-export: refactor composite functions now live in indicators.composite."""

from indicators.composite import (
    load_refactor_config,
    g_pers_refactor,
    compute_opportunity_refactor,
    compute_gate_refactor,
    compute_composite_score_refactor,
    composite_refactor_from_config,
)

__all__ = [
    "load_refactor_config",
    "g_pers_refactor",
    "compute_opportunity_refactor",
    "compute_gate_refactor",
    "compute_composite_score_refactor",
    "composite_refactor_from_config",
]
