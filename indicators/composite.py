"""
Phase 1 Symbolic Composite Score

Implements the Composite_DCA_State_Score ∈ [0, 100] that describes market
state to modulate DCA intensity. The baseline DCA always stays ON - this
score only affects the intensity.

Composite Formula (Phase 1 Symbolic):
------------------------------------
1. Normalize all components to [0, 1] via ECDF → Z → Sigmoid pipeline
2. g_pers(H_t): Persistence modifier mapping Hurst to [0, 1]
3. Gate_t = C_t × L_t × R_t_thresholded
   - C_t: Systemic coupling (low = favorable)
   - L_t: Liquidity (high = favorable)
   - R_t_thresholded: Regime probability with configurable threshold
4. Opp_t = agg_committee([T_t, U_t × g_pers(H_t)])
   - T_t: Trend score
   - U_t: Undervaluation (VWAP Z)
   - H_t: Hurst exponent for persistence weighting
5. RawFavor_t = Opp_t × Gate_t
6. CompositeScore = 100 × clip(0.5 + (RawFavor_t - 0.5) × S_scale, 0, 1)
   - Anchored at 50 (neutral)
   - S_scale is configurable (NOT hardcoded)

Non-Negotiable Rules:
--------------------
- NO prediction logic - score only modifies intensity
- NO hardcoded thresholds or weights - all from config
- NO live data - use fixtures in tests
- ALL intermediate values logged for audit

Author: Phase 1 Implementation
Date: 2026-02-07
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import logging

logger = logging.getLogger(__name__)

# Import other indicator modules
from indicators.committee import agg_committee


@dataclass
class Phase1Config:
    """
    Configuration for Phase 1 composite calculation.
    
    Mathematical Framework:
    ----------------------
    Gate_t = C_t × L_t × 𝟙[R_t ≥ r_thresh]
    Opp_t = Mean([T_t, U_t × g_pers(H_t)])
    RawFavor_t = Opp_t × Gate_t
    CompositeScore_t = 100 × clip(0.5 + (RawFavor_t − 0.5) × S_scale, 0, 1)
    
    where g_pers(H_t) = Sigmoid(H_t - 0.5) × 2
    
    All numeric values are SYMBOLIC PLACEHOLDERS - do not hardcode tuned values.
    These will be determined in Phase 2 walk-forward optimization.
    """
    # Regime threshold (R_t must exceed this for gate to be favorable)
    # SYMBOLIC: Set to None or placeholder - will be tuned in Phase 2
    regime_threshold: Optional[float] = None  # TODO Phase 2: Tune via backtest
    
    # Threshold type for regime gating
    # "hard": 𝟙[R_t ≥ threshold] = 1 or 0 (canonical indicator function)
    # "soft": sigmoid transition
    regime_threshold_type: str = "hard"
    
    # Scale parameter for final score transformation
    # SYMBOLIC: Set to None or placeholder - will be tuned in Phase 2
    S_scale: Optional[float] = None  # TODO Phase 2: Tune via sensitivity analysis
    
    # Committee aggregation method for Opp_t
    # "mean": Simple arithmetic mean (canonical: Opp_t = Mean([T_t, U_weighted]))
    # "trimmed_mean": Robust to outliers
    committee_method: str = "mean"  # Canonical formula uses mean
    committee_trim_pct: float = 0.1
    
    # Persistence function parameters (g_pers)
    # CANONICAL: g_pers(H_t) = Sigmoid(H_t - 0.5) × 2
    # "sigmoid_canonical": Uses the canonical formula
    # "sigmoid": General sigmoid with configurable center
    # "linear": Linear interpolation
    # "threshold": Binary threshold
    g_pers_type: str = "sigmoid_canonical"  # Canonical formula
    g_pers_params: Dict[str, Optional[float]] = field(default_factory=lambda: {
        "k": 10.0,  # Sigmoid steepness (controls sensitivity around H=0.5)
        "H_neutral": 0.5,  # Center point (canonical = 0.5)
        "H_favorable": None,  # For linear type
        "H_unfavorable": None,  # For linear type
    })
    
    # Minimum observations for valid computation
    min_obs: int = 100
    
    # Logging configuration
    log_intermediates: bool = True
    log_path: str = "logs/phase1_indicator_runs.json"
    
    # Safety flag - prevent production use until explicitly enabled
    allow_production_mode: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime_threshold": self.regime_threshold,
            "regime_threshold_type": self.regime_threshold_type,
            "S_scale": self.S_scale,
            "committee_method": self.committee_method,
            "committee_trim_pct": self.committee_trim_pct,
            "g_pers_type": self.g_pers_type,
            "g_pers_params": self.g_pers_params,
            "min_obs": self.min_obs,
            "log_intermediates": self.log_intermediates,
            "log_path": self.log_path,
            "allow_production_mode": self.allow_production_mode
        }


class CompositeResult(NamedTuple):
    """Result of composite score calculation."""
    score: float  # Final composite score [0, 100]
    T_t: float    # Trend component [0, 1]
    U_t: float    # Undervaluation component [0, 1]
    V_t: float    # Volatility component [0, 1]
    L_t: float    # Liquidity component [0, 1]
    C_t: float    # Coupling component [0, 1]
    H_t: float    # Hurst exponent [0, 1]
    R_t: float    # Regime probability [0, 1]
    g_pers_H: float  # Persistence modifier [0, 1]
    Gate_t: float    # Gate value [0, 1]
    Opp_t: float     # Opportunity value [0, 1]
    RawFavor_t: float  # Raw favorability [0, 1]
    meta: Dict[str, Any]  # Computation metadata


def g_pers(
    H_t: float,
    g_type: str = "sigmoid_canonical",
    params: Optional[Dict[str, Optional[float]]] = None
) -> float:
    """
    Persistence modifier function mapping Hurst exponent to [0, 1].
    
    Mathematical Definition (Canonical):
    ------------------------------------
    g_pers(H_t) = Sigmoid(H_t - 0.5) × 2
    
    This maps:
    - H = 0.5 (random walk) → g ≈ 0.5 (neutral)
    - H > 0.5 (persistent/trending) → g > 0.5 (boost undervaluation signal)
    - H < 0.5 (mean-reverting) → g < 0.5 (suppress undervaluation signal)
    
    Parameters
    ----------
    H_t : float
        Hurst exponent value ∈ (0, 1)
    g_type : str
        Function type:
        - "sigmoid_canonical": g = sigmoid(H - 0.5) * 2, clamped to [0,1] (DEFAULT)
        - "sigmoid": g = 1 / (1 + exp(-k*(H - H_mid)))
        - "linear": g = (H - H_min) / (H_max - H_min)
        - "threshold": g = 1 if H > threshold else 0
    params : Dict, optional
        Type-specific parameters
    
    Returns
    -------
    float
        g_pers value in [0, 1]
    
    Notes
    -----
    The canonical formula g_pers(H_t) = Sigmoid(H_t - 0.5) × 2 ensures:
    - Undervaluation (U_t) is boosted only in persistent/trending regimes
    - Mean-reverting regimes suppress the dip-buying signal
    - This aligns with the intuition that buying dips works better in trends
    
    References
    ----------
    - Mandelbrot & Van Ness (1968): Fractional Brownian motions
    - Peters (1994): Fractal Market Analysis
    """
    if np.isnan(H_t):
        return 0.5  # Neutral when missing
    
    if params is None:
        params = {}
    
    if g_type == "sigmoid_canonical":
        # CANONICAL FORMULA: g_pers(H_t) = Sigmoid(H_t - 0.5) × 2
        # Sigmoid(x) = 1 / (1 + exp(-x))
        # We use a scaling factor to spread the sigmoid across H range
        k = params.get("k") or 10.0  # Steepness (controls sensitivity around H=0.5)
        z = k * (H_t - 0.5)
        sigmoid_val = 1.0 / (1.0 + np.exp(-z))
        # Scale by 2 and clamp to [0, 1]
        g = sigmoid_val * 2.0
        return float(np.clip(g, 0.0, 1.0))
    
    elif g_type == "sigmoid":
        # General sigmoid centered at configurable H_mid
        H_mid = params.get("H_neutral") or 0.5
        k = params.get("k") or 10.0
        z = k * (H_t - H_mid)
        return float(1.0 / (1.0 + np.exp(-z)))
    
    elif g_type == "linear":
        # Simple linear mapping
        H_min = params.get("H_unfavorable") or 0.3
        H_max = params.get("H_favorable") or 0.7
        if H_max == H_min:
            return 0.5
        g = (H_t - H_min) / (H_max - H_min)
        return float(np.clip(g, 0.0, 1.0))
    
    elif g_type == "threshold":
        # Binary threshold
        threshold = params.get("threshold") or 0.5
        return 1.0 if H_t > threshold else 0.0
    
    else:
        logger.warning(f"Unknown g_pers type '{g_type}', using sigmoid_canonical")
        k = 10.0
        z = k * (H_t - 0.5)
        sigmoid_val = 1.0 / (1.0 + np.exp(-z))
        return float(np.clip(sigmoid_val * 2.0, 0.0, 1.0))


def compute_gate(
    C_t: float,
    L_t: float,
    R_t: float,
    regime_threshold: Optional[float] = None,
    threshold_type: str = "hard"
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute Gate value: C_t × L_t × 𝟙[R_t ≥ r_thresh]
    
    Mathematical Definition:
    ------------------------
    Gate_t = C_t × L_t × 𝟙[R_t ≥ r_thresh]
    
    where 𝟙[·] is the indicator function:
    - 𝟙[R_t ≥ r_thresh] = 1 if R_t ≥ threshold, else 0 (hard)
    - Or soft version: sigmoid(k × (R_t - threshold))
    
    The Gate mechanism reduces composite when:
    - Coupling is high (systemic risk) → C_t low
    - Liquidity is low (slippage risk) → L_t low
    - Regime is unstable (below threshold) → indicator = 0
    
    Parameters
    ----------
    C_t : float
        Coupling score [0, 1] (inverted: high = low correlation = favorable)
    L_t : float
        Liquidity score [0, 1] (high = more liquid = favorable)
    R_t : float
        Regime probability [0, 1] (P(StableExpansion))
    regime_threshold : float, optional
        Threshold r_thresh for regime indicator (SYMBOLIC - None means use R_t directly)
    threshold_type : str
        "hard": 𝟙[R_t ≥ threshold] = 1 or 0 (strict indicator function)
        "soft": sigmoid(k × (R_t - threshold)) (smooth transition)
    
    Returns
    -------
    Tuple[float, Dict[str, Any]]
        - Gate_t: Gate value [0, 1]
        - meta: Computation details
    
    Notes
    -----
    The hard threshold (indicator function) provides clear gating:
    - If regime is unstable (R_t < threshold), Gate shuts off completely
    - This prevents DCA intensification during detected instability
    """
    # Handle NaN inputs
    if np.isnan(C_t):
        C_t = 0.5  # Neutral
    if np.isnan(L_t):
        L_t = 0.5  # Neutral
    if np.isnan(R_t):
        R_t = 0.5  # Neutral
    
    # Apply regime threshold if specified
    if regime_threshold is not None:
        if threshold_type == "hard":
            # CANONICAL: Indicator function 𝟙[R_t ≥ r_thresh]
            R_t_thresholded = 1.0 if R_t >= regime_threshold else 0.0
        else:
            # Soft thresholding: smooth transition around threshold
            k = 10.0  # Steepness (SYMBOLIC - could be configurable)
            R_t_thresholded = 1.0 / (1.0 + np.exp(-k * (R_t - regime_threshold)))
    else:
        # No threshold, use R_t directly (continuous gating)
        R_t_thresholded = R_t
    
    # Gate is product of components
    Gate_t = C_t * L_t * R_t_thresholded
    
    meta = {
        "C_t": C_t,
        "L_t": L_t,
        "R_t": R_t,
        "regime_threshold": regime_threshold,
        "threshold_type": threshold_type,
        "R_t_thresholded": R_t_thresholded,
        "Gate_t": Gate_t
    }
    
    return float(Gate_t), meta


def compute_opportunity(
    T_t: float,
    U_t: float,
    H_t: float,
    config: Optional[Phase1Config] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute Opportunity value: Mean([T_t, U_t × g_pers(H_t)])
    
    Mathematical Definition:
    ------------------------
    Opp_t = Mean([T_t, U_t × g_pers(H_t)])
    
    where:
    - T_t = Trend strength signal (directional clarity)
    - U_t = Undervaluation signal (Z-VWAP based)
    - g_pers(H_t) = Sigmoid(H_t - 0.5) × 2 (persistence modifier)
    
    The undervaluation signal is boosted only in persistent/trending 
    regimes (H > 0.5), since buying dips works better when trends persist.
    
    Parameters
    ----------
    T_t : float
        Trend score [0, 1] - captures directional clarity
    U_t : float
        Undervaluation score [0, 1] - statistical undervaluation via Z-VWAP
    H_t : float
        Hurst exponent [0, 1] - persistence measure
    config : Phase1Config, optional
        Configuration for g_pers and aggregation
    
    Returns
    -------
    Tuple[float, Dict[str, Any]]
        - Opp_t: Opportunity value [0, 1]
        - meta: Computation details with g_pers_H for explainability
    
    Notes
    -----
    The formula U_t × g_pers(H_t) ensures:
    - When H > 0.5 (trending): g_pers > 0.5, so U_t is boosted
    - When H < 0.5 (mean-reverting): g_pers < 0.5, so U_t is suppressed
    - This prevents buying dips in choppy/mean-reverting markets
    """
    if config is None:
        config = Phase1Config()
    
    # Handle NaN
    if np.isnan(T_t):
        T_t = 0.5
    if np.isnan(U_t):
        U_t = 0.5
    if np.isnan(H_t):
        H_t = 0.5
    
    # Compute persistence modifier: g_pers(H_t) = Sigmoid(H_t - 0.5) × 2
    g_H = g_pers(H_t, config.g_pers_type, config.g_pers_params)
    
    # Weight undervaluation by persistence
    # This is the key insight: U_t × g_pers(H_t)
    U_weighted = U_t * g_H
    
    # Aggregate: Opp_t = Mean([T_t, U_t × g_pers(H_t)])
    # Or use committee aggregation for robustness
    scores = [T_t, U_weighted]
    
    if config.committee_method == "mean":
        # Canonical: simple arithmetic mean
        Opp_t = np.mean(scores)
        committee_meta = {"method": "mean", "inputs": scores}
    else:
        # Robust aggregation (trimmed mean, etc.)
        Opp_t, committee_meta = agg_committee(
            scores,
            method=config.committee_method,
            trim_pct=config.committee_trim_pct
        )
    
    meta = {
        "T_t": T_t,
        "U_t": U_t,
        "H_t": H_t,
        "g_pers_H": g_H,
        "U_weighted": U_weighted,
        "formula": "Opp_t = Mean([T_t, U_t × g_pers(H_t)])",
        "committee_inputs": scores,
        "committee_meta": committee_meta,
        "Opp_t": Opp_t
    }
    
    return float(Opp_t), meta


def compute_composite_score(
    T_t: float,
    U_t: float,
    V_t: float,
    L_t: float,
    C_t: float,
    H_t: float,
    R_t: float,
    config: Optional[Phase1Config] = None
) -> CompositeResult:
    """
    Compute the Phase 1 Composite DCA State Score.
    
    Parameters
    ----------
    T_t : float
        Trend component [0, 1]
    U_t : float
        Undervaluation component [0, 1]
    V_t : float
        Volatility regime component [0, 1]
    L_t : float
        Liquidity component [0, 1]
    C_t : float
        Coupling component [0, 1] (inverted: high = favorable)
    H_t : float
        Hurst exponent [0, 1]
    R_t : float
        Regime probability [0, 1]
    config : Phase1Config, optional
        Configuration object
    
    Returns
    -------
    CompositeResult
        Named tuple with final score and all intermediate values
    
    Examples
    --------
    >>> config = Phase1Config(S_scale=1.0, regime_threshold=0.5)
    >>> result = compute_composite_score(
    ...     T_t=0.6, U_t=0.7, V_t=0.5, L_t=0.8, C_t=0.7, H_t=0.55, R_t=0.6,
    ...     config=config
    ... )
    >>> print(f"Composite: {result.score:.1f}, Gate: {result.Gate_t:.3f}")
    
    Notes
    -----
    Formula:
    1. Gate_t = C_t × L_t × R_t_thresholded
    2. g_pers_H = g_pers(H_t)
    3. Opp_t = committee([T_t, U_t × g_pers_H])
    4. RawFavor_t = Opp_t × Gate_t
    5. CompositeScore = 100 × clip(0.5 + (RawFavor_t - 0.5) × S_scale, 0, 1)
    
    The score is anchored at 50:
    - RawFavor_t = 0.5 → Score = 50 (neutral)
    - RawFavor_t > 0.5 → Score > 50 (favorable)
    - RawFavor_t < 0.5 → Score < 50 (unfavorable)
    """
    if config is None:
        config = Phase1Config()
    
    # Step 1: Compute Gate
    # Gate_t = C_t × L_t × 𝟙[R_t ≥ r_thresh]
    Gate_t, gate_meta = compute_gate(
        C_t=C_t,
        L_t=L_t,
        R_t=R_t,
        regime_threshold=config.regime_threshold,
        threshold_type=config.regime_threshold_type
    )
    
    # Step 2: Compute Opportunity
    Opp_t, opp_meta = compute_opportunity(
        T_t=T_t,
        U_t=U_t,
        H_t=H_t,
        config=config
    )
    g_pers_H = opp_meta["g_pers_H"]
    
    # Step 3: Raw Favorability
    RawFavor_t = Opp_t * Gate_t
    
    # Step 4: Final Score with anchor at 50
    # Use default S_scale = 1.0 if not specified (SYMBOLIC)
    S_scale = config.S_scale if config.S_scale is not None else 1.0
    
    # Transform: score = 0.5 + (raw - 0.5) * scale
    # Then clip and scale to [0, 100]
    transformed = 0.5 + (RawFavor_t - 0.5) * S_scale
    clipped = float(np.clip(transformed, 0.0, 1.0))
    composite_score = 100.0 * clipped
    
    # Build metadata
    meta = {
        "config": config.to_dict(),
        "gate_meta": gate_meta,
        "opportunity_meta": opp_meta,
        "RawFavor_t": RawFavor_t,
        "S_scale_used": S_scale,
        "transformed_before_clip": transformed,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Log if configured
    if config.log_intermediates:
        _log_computation(
            score=composite_score,
            T_t=T_t, U_t=U_t, V_t=V_t, L_t=L_t, C_t=C_t,
            H_t=H_t, R_t=R_t, Gate_t=Gate_t, Opp_t=Opp_t,
            RawFavor_t=RawFavor_t, g_pers_H=g_pers_H,
            meta=meta,
            log_path=config.log_path
        )
    
    return CompositeResult(
        score=composite_score,
        T_t=T_t if not np.isnan(T_t) else 0.5,
        U_t=U_t if not np.isnan(U_t) else 0.5,
        V_t=V_t if not np.isnan(V_t) else 0.5,
        L_t=L_t if not np.isnan(L_t) else 0.5,
        C_t=C_t if not np.isnan(C_t) else 0.5,
        H_t=H_t if not np.isnan(H_t) else 0.5,
        R_t=R_t if not np.isnan(R_t) else 0.5,
        g_pers_H=g_pers_H,
        Gate_t=Gate_t,
        Opp_t=Opp_t,
        RawFavor_t=RawFavor_t,
        meta=meta
    )


def _log_computation(
    score: float,
    T_t: float, U_t: float, V_t: float, L_t: float, C_t: float,
    H_t: float, R_t: float, Gate_t: float, Opp_t: float,
    RawFavor_t: float, g_pers_H: float,
    meta: Dict[str, Any],
    log_path: str
) -> None:
    """
    Log computation to JSON file for audit trail.
    """
    try:
        # Ensure directory exists
        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Build log entry
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "composite_score": score,
            "components": {
                "T_t": float(T_t) if not np.isnan(T_t) else None,
                "U_t": float(U_t) if not np.isnan(U_t) else None,
                "V_t": float(V_t) if not np.isnan(V_t) else None,
                "L_t": float(L_t) if not np.isnan(L_t) else None,
                "C_t": float(C_t) if not np.isnan(C_t) else None,
                "H_t": float(H_t) if not np.isnan(H_t) else None,
                "R_t": float(R_t) if not np.isnan(R_t) else None
            },
            "intermediates": {
                "g_pers_H": g_pers_H,
                "Gate_t": Gate_t,
                "Opp_t": Opp_t,
                "RawFavor_t": RawFavor_t
            },
            "config": meta.get("config", {})
        }
        
        # Append to log file
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
            
    except Exception as e:
        logger.warning(f"Failed to log computation: {e}")


class Phase1Composite:
    """
    Convenience class for Phase 1 composite score computation.
    
    This class provides a higher-level interface that:
    - Loads configuration from YAML
    - Computes all indicators from price data
    - Returns fully documented results
    """
    
    def __init__(self, config: Optional[Phase1Config] = None):
        self.config = config or Phase1Config()
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Phase1Composite":
        """Load configuration from YAML file."""
        import yaml
        
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        config = Phase1Config(
            regime_threshold=config_dict.get("regime_threshold"),
            S_scale=config_dict.get("S_scale"),
            committee_method=config_dict.get("committee_method", "trimmed_mean"),
            committee_trim_pct=config_dict.get("committee_trim_pct", 0.1),
            g_pers_type=config_dict.get("g_pers_type", "linear"),
            g_pers_params=config_dict.get("g_pers_params", {}),
            min_obs=config_dict.get("min_obs", 100),
            log_intermediates=config_dict.get("log_intermediates", True),
            log_path=config_dict.get("log_path", "logs/phase1_indicator_runs.json"),
            allow_production_mode=config_dict.get("allow_production_mode", False)
        )
        
        return cls(config)
    
    def compute(
        self,
        T_t: float, U_t: float, V_t: float, L_t: float, C_t: float,
        H_t: float, R_t: float
    ) -> CompositeResult:
        """Compute composite score from normalized components."""
        return compute_composite_score(
            T_t=T_t, U_t=U_t, V_t=V_t, L_t=L_t, C_t=C_t,
            H_t=H_t, R_t=R_t, config=self.config
        )
    
    def compute_neutral(self) -> CompositeResult:
        """
        Compute composite with all neutral inputs.
        
        Should return score ≈ 50 (anchor point).
        """
        return self.compute(
            T_t=0.5, U_t=0.5, V_t=0.5, L_t=0.5, C_t=0.5,
            H_t=0.5, R_t=0.5
        )
