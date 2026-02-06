"""
Phase 1 Market State Indicator Engine

This package implements the Phase 1 indicators for the Composite DCA State Score.
The composite describes market state (not predict returns) to modulate DCA intensity.

Modules:
--------
- hurst: Wavelet-based Hurst exponent estimator (persistence/mean-reversion)
- hmm_regime: HMM-based regime detection (P(StableExpansion))
- vwap_z: VWAP-based valuation Z-score
- volatility: Realized volatility and percentile calculations
- liquidity: Amihud illiquidity measure
- coupling: Systemic coupling via shrinkage covariance
- normalization: ECDF → Z → Sigmoid pipeline
- committee: Robust aggregation (trimmed mean)
- composite: Phase 1 symbolic composite score

Non-Negotiable Rules:
--------------------
1. No prediction: Score only modifies intensity, never stops baseline DCA
2. No live data: Use fixtures when no provider keys present
3. No numeric finalization: All thresholds/weights are symbolic placeholders
4. Reproducibility: All functions must be deterministic with seeded RNG
5. Explainability: All functions return (value, metadata) tuples

Author: Phase 1 Implementation
Date: 2026-02-07
"""

from indicators.hurst import estimate_hurst
from indicators.hmm_regime import infer_regime_prob
from indicators.vwap_z import compute_vwap_z
from indicators.volatility import realized_vol, volatility_percentile
from indicators.liquidity import amihud_illiquidity
from indicators.coupling import systemic_coupling
from indicators.normalization import (
    expanding_percentile,
    percentile_to_z,
    z_to_sigmoid,
    polarity_align,
    normalize_to_score
)
from indicators.committee import agg_committee
from indicators.composite import compute_composite_score, Phase1Composite

__all__ = [
    # Hurst
    "estimate_hurst",
    # HMM
    "infer_regime_prob",
    # VWAP
    "compute_vwap_z",
    # Volatility
    "realized_vol",
    "volatility_percentile",
    # Liquidity
    "amihud_illiquidity",
    # Coupling
    "systemic_coupling",
    # Normalization
    "expanding_percentile",
    "percentile_to_z",
    "z_to_sigmoid",
    "polarity_align",
    "normalize_to_score",
    # Committee
    "agg_committee",
    # Composite
    "compute_composite_score",
    "Phase1Composite",
]

__version__ = "0.1.0-phase1"
