"""
Normalization Pipeline

Implements the expanding ECDF → inverse-normal → sigmoid pipeline for
normalizing indicator values to [0, 1] scores.

Pipeline:
---------
1. expanding_percentile: Convert raw value to percentile using historical ECDF
2. percentile_to_z: Convert percentile to Z-score (inverse normal CDF)
3. z_to_sigmoid: Apply sigmoid transform for bounded output
4. polarity_align: Ensure higher score = more favorable

This approach ensures:
- No lookahead bias (expanding window only uses past data)
- Comparable scores across different indicators
- Bounded output [0, 1]
- Interpretable intermediate values

Author: Phase 1 Implementation
Date: 2026-02-07
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)


def expanding_percentile(
    series: np.ndarray,
    min_obs: int = 100
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute expanding percentile rank (ECDF) without lookahead.
    
    At each time t, computes the percentile rank of series[t] relative
    to all observations from 0 to t-1 (exclusive of current).
    
    Parameters
    ----------
    series : np.ndarray
        Input time series
    min_obs : int
        Minimum observations before computing percentile (default: 100)
    
    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        - pct_t: Percentile values [0, 1] (NaN for warmup period)
        - meta: Metadata with sample sizes per observation
    
    Examples
    --------
    >>> import numpy as np
    >>> values = np.random.randn(500)
    >>> pct_t, meta = expanding_percentile(values, min_obs=100)
    >>> print(f"Percentile at t=200: {pct_t[200]:.3f}")
    
    Notes
    -----
    - No lookahead: percentile[t] uses only series[0:t]
    - Returns raw percentile in [0, 1], not [0, 100]
    - NaN returned for first min_obs observations
    """
    series = np.asarray(series, dtype=np.float64)
    n = len(series)
    
    pct_t = np.full(n, np.nan)
    sample_sizes = np.full(n, 0, dtype=int)
    
    for t in range(min_obs, n):
        current_value = series[t]
        if np.isnan(current_value):
            continue
        
        # Historical values (excluding current)
        historical = series[:t]
        valid_hist = historical[~np.isnan(historical)]
        
        if len(valid_hist) < min_obs:
            continue
        
        sample_sizes[t] = len(valid_hist)
        
        # Percentile rank: fraction of historical values <= current
        # Add small tie-breaking for stability
        pct = np.sum(valid_hist <= current_value) / len(valid_hist)
        
        # Clip to avoid exact 0 or 1 (causes inf in inverse normal)
        pct_t[t] = np.clip(pct, 0.001, 0.999)
    
    meta = {
        "min_obs": min_obs,
        "n_total": n,
        "method": "expanding_ecdf",
        "seed": None,
        "notes": f"Expanding ECDF percentile with min_obs={min_obs}"
    }
    
    return pct_t, meta


def percentile_to_z(
    percentile: np.ndarray
) -> np.ndarray:
    """
    Convert percentile to Z-score via inverse normal CDF.
    
    This transforms a uniform [0, 1] percentile to a standard normal
    Z-score, which can then be bounded via sigmoid.
    
    Parameters
    ----------
    percentile : np.ndarray
        Percentile values in [0, 1]
    
    Returns
    -------
    np.ndarray
        Z-scores (standard normal)
    
    Notes
    -----
    - Percentiles must be in (0, 1), not exactly 0 or 1
    - Uses scipy.stats.norm.ppf (probit function)
    """
    percentile = np.asarray(percentile, dtype=np.float64)
    
    # Clip to avoid inf
    safe_pct = np.clip(percentile, 0.001, 0.999)
    
    # Inverse normal CDF
    z = stats.norm.ppf(safe_pct)
    
    # Preserve NaN
    z = np.where(np.isnan(percentile), np.nan, z)
    
    return z


def z_to_sigmoid(
    z: np.ndarray,
    k: float = 1.0
) -> np.ndarray:
    """
    Apply sigmoid transform to Z-scores.
    
    sigmoid(z) = 1 / (1 + exp(-k*z))
    
    Parameters
    ----------
    z : np.ndarray
        Z-scores
    k : float
        Steepness parameter (default: 1.0)
        - k > 1: Steeper transition around z=0
        - k < 1: Gentler transition
    
    Returns
    -------
    np.ndarray
        Sigmoid values in (0, 1)
    
    Notes
    -----
    - Z=0 maps to sigmoid=0.5 (neutral)
    - Z>0 maps to sigmoid>0.5
    - Z<0 maps to sigmoid<0.5
    """
    z = np.asarray(z, dtype=np.float64)
    
    # Clip to avoid overflow
    safe_z = np.clip(z * k, -500, 500)
    
    sigmoid = 1.0 / (1.0 + np.exp(-safe_z))
    
    # Preserve NaN
    sigmoid = np.where(np.isnan(z), np.nan, sigmoid)
    
    return sigmoid


def polarity_align(
    score: np.ndarray,
    higher_is_favorable: bool = True
) -> np.ndarray:
    """
    Align polarity so higher score = more favorable.
    
    Parameters
    ----------
    score : np.ndarray
        Input score in [0, 1]
    higher_is_favorable : bool
        If True (default), score unchanged
        If False, invert: 1 - score
    
    Returns
    -------
    np.ndarray
        Polarity-aligned score in [0, 1]
    
    Notes
    -----
    Examples of indicators needing inversion (higher_is_favorable=False):
    - Illiquidity (high illiquidity = unfavorable)
    - Systemic coupling (high coupling = unfavorable in crisis)
    - Volatility percentile (high vol = potentially unfavorable)
    
    Examples already aligned (higher_is_favorable=True):
    - Undervaluation score (high = more undervalued = favorable)
    - Trend score when in uptrend
    """
    score = np.asarray(score, dtype=np.float64)
    
    if higher_is_favorable:
        return score
    else:
        return 1.0 - score


def normalize_to_score(
    raw_values: np.ndarray,
    min_obs: int = 100,
    k: float = 1.0,
    higher_is_favorable: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Full normalization pipeline: ECDF → Z → Sigmoid → Polarity.
    
    This is the recommended function for normalizing any raw indicator
    to a [0, 1] score suitable for the composite calculation.
    
    Parameters
    ----------
    raw_values : np.ndarray
        Raw indicator values
    min_obs : int
        Minimum observations for ECDF (default: 100)
    k : float
        Sigmoid steepness (default: 1.0)
    higher_is_favorable : bool
        Whether higher raw values are favorable (default: True)
    
    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        - score_t: Normalized score in [0, 1]
        - meta: Metadata including intermediate values
    
    Examples
    --------
    >>> import numpy as np
    >>> # Example: Hurst exponent (higher H = more persistent = favorable)
    >>> H_raw = np.random.uniform(0.3, 0.7, 500)
    >>> score, meta = normalize_to_score(H_raw, higher_is_favorable=True)
    >>> print(f"Score range: [{np.nanmin(score):.3f}, {np.nanmax(score):.3f}]")
    """
    # Step 1: Expanding percentile
    pct_t, pct_meta = expanding_percentile(raw_values, min_obs=min_obs)
    
    # Step 2: Percentile to Z
    z_t = percentile_to_z(pct_t)
    
    # Step 3: Z to sigmoid
    sigmoid_t = z_to_sigmoid(z_t, k=k)
    
    # Step 4: Polarity alignment
    score_t = polarity_align(sigmoid_t, higher_is_favorable=higher_is_favorable)
    
    meta = {
        "min_obs": min_obs,
        "sigmoid_k": k,
        "higher_is_favorable": higher_is_favorable,
        "n_obs": len(raw_values),
        "method": "ecdf_z_sigmoid",
        "seed": None,
        "notes": f"Full normalization: ECDF({min_obs}) → Z → sigmoid(k={k}) → polarity({higher_is_favorable})"
    }
    
    return score_t, meta


def batch_normalize(
    indicators: Dict[str, np.ndarray],
    config: Optional[Dict[str, Dict[str, Any]]] = None,
    default_min_obs: int = 100,
    default_k: float = 1.0
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
    """
    Normalize multiple indicators in batch.
    
    Parameters
    ----------
    indicators : Dict[str, np.ndarray]
        Dict of indicator name → raw values
    config : Dict[str, Dict], optional
        Per-indicator config with keys: min_obs, k, higher_is_favorable
    default_min_obs : int
        Default minimum observations
    default_k : float
        Default sigmoid steepness
    
    Returns
    -------
    Tuple[Dict[str, np.ndarray], Dict[str, Dict]]
        - normalized: Dict of indicator name → normalized score
        - metas: Dict of indicator name → metadata
    
    Examples
    --------
    >>> indicators = {
    ...     "hurst": np.random.uniform(0.3, 0.7, 500),
    ...     "vwap_z": np.random.randn(500),
    ...     "volatility": np.random.uniform(0.1, 0.4, 500)
    ... }
    >>> config = {
    ...     "hurst": {"higher_is_favorable": True},
    ...     "vwap_z": {"higher_is_favorable": False},  # Negative Z is favorable
    ...     "volatility": {"higher_is_favorable": False}  # Low vol is favorable
    ... }
    >>> normalized, metas = batch_normalize(indicators, config)
    """
    if config is None:
        config = {}
    
    normalized = {}
    metas = {}
    
    for name, raw_values in indicators.items():
        ind_config = config.get(name, {})
        
        score, meta = normalize_to_score(
            raw_values,
            min_obs=ind_config.get("min_obs", default_min_obs),
            k=ind_config.get("k", default_k),
            higher_is_favorable=ind_config.get("higher_is_favorable", True)
        )
        
        normalized[name] = score
        metas[name] = meta
    
    return normalized, metas
