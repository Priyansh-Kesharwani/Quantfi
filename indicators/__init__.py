"""
Market State Indicator Engine - Phase 2

Complete indicator suite for the Composite DCA State Score.

Components:
- T_t: Trend strength (trend.py)
- U_t: Undervaluation (undervaluation.py)  
- H_t: Hurst exponent (hurst.py)
- V_t: Volatility regime (volatility.py)
- L_t: Liquidity (liquidity.py)
- C_t: Systemic coupling (coupling.py)
- R_t: Regime probability (hmm_regime.py)

Utilities:
- normalization.py: ECDF -> Z -> Sigmoid pipeline
- committee.py: Robust aggregation
- composite.py: Final composite score calculation
- indicator_engine.py: Unified indicator computation

Non-Negotiable Rules:
--------------------
1. No prediction: Score only modifies intensity, never stops baseline DCA
2. No live data: Use fixtures when no provider keys present
3. No numeric finalization: All thresholds/weights are symbolic placeholders
4. Reproducibility: All functions must be deterministic with seeded RNG
5. Explainability: All functions return (value, metadata) tuples

Author: Phase 1/2 Implementation
Date: 2026-02-07
"""

# Hurst exponent
from indicators.hurst import estimate_hurst, hurst_exponent

# HMM Regime
from indicators.hmm_regime import infer_regime_prob, regime_probability, HMMRegimeConfig

# VWAP-Z
from indicators.vwap_z import compute_vwap_z

# Volatility
from indicators.volatility import realized_vol, volatility_percentile, volatility_regime_score

# Liquidity
from indicators.liquidity import amihud_illiquidity, liquidity_score

# Coupling
from indicators.coupling import coupling_score, systemic_coupling

# Normalization
from indicators.normalization import (
    expanding_percentile,
    percentile_to_z,
    z_to_sigmoid,
    polarity_align,
    normalize_to_score,
    batch_normalize
)

# Committee aggregation
from indicators.committee import agg_committee

# Composite
from indicators.composite import (
    g_pers, compute_gate, compute_opportunity, compute_composite_score,
    Phase1Config, CompositeResult, Phase1Composite
)

# Trend (Phase 2)
from indicators.trend import (
    trend_strength_score, adx_indicator, macd_histogram, ema_slope
)

# Undervaluation (Phase 2)
from indicators.undervaluation import (
    undervaluation_score, price_vwap_zscore, drawdown_score
)

# Unified Engine (Phase 2)
from indicators.indicator_engine import (
    IndicatorEngine, IndicatorConfig, IndicatorResult, compute_all_indicators
)

__all__ = [
    # Hurst
    "estimate_hurst",
    "hurst_exponent",
    # HMM Regime
    "infer_regime_prob",
    "regime_probability",
    "HMMRegimeConfig",
    # VWAP
    "compute_vwap_z",
    # Volatility
    "realized_vol",
    "volatility_percentile",
    "volatility_regime_score",
    # Liquidity
    "amihud_illiquidity",
    "liquidity_score",
    # Coupling
    "systemic_coupling",
    "coupling_score",
    # Normalization
    "expanding_percentile",
    "percentile_to_z",
    "z_to_sigmoid",
    "polarity_align",
    "normalize_to_score",
    "batch_normalize",
    # Committee
    "agg_committee",
    # Composite
    "g_pers",
    "compute_gate",
    "compute_opportunity",
    "compute_composite_score",
    "Phase1Config",
    "CompositeResult",
    "Phase1Composite",
    # Trend
    "trend_strength_score",
    "adx_indicator",
    "macd_histogram",
    "ema_slope",
    # Undervaluation
    "undervaluation_score",
    "price_vwap_zscore",
    "drawdown_score",
    # Engine
    "IndicatorEngine",
    "IndicatorConfig",
    "IndicatorResult",
    "compute_all_indicators",
]

__version__ = "0.2.0-phase2"
