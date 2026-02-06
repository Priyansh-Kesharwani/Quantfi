"""
Committee Aggregation

Robust aggregation of multiple scores using trimmed mean and other
consensus methods. This reduces the influence of outliers and provides
a more stable composite signal.

Methods:
--------
- trimmed_mean: Remove top/bottom percentiles before averaging
- winsorized_mean: Cap extreme values instead of removing
- median: Most robust but loses information
- weighted_mean: Custom weights per score

References:
-----------
1. Huber, P.J. (1981). "Robust Statistics"
2. Hampel, F.R. et al. (1986). "Robust Statistics: The Approach Based
   on Influence Functions"

Author: Phase 1 Implementation
Date: 2026-02-07
"""

import numpy as np
from typing import Tuple, Dict, Any, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


def _trimmed_mean(
    values: np.ndarray,
    trim_pct: float = 0.1
) -> float:
    """
    Compute trimmed mean.
    
    Removes top and bottom trim_pct of values before averaging.
    
    Parameters
    ----------
    values : np.ndarray
        Input values
    trim_pct : float
        Fraction to trim from each end (default: 0.1 = 10%)
    
    Returns
    -------
    float
        Trimmed mean
    """
    values = np.asarray(values, dtype=np.float64)
    valid = values[~np.isnan(values)]
    
    if len(valid) == 0:
        return np.nan
    
    if len(valid) <= 2:
        return np.mean(valid)
    
    # Sort and trim
    sorted_vals = np.sort(valid)
    n = len(sorted_vals)
    
    # Number to trim from each end
    trim_count = int(np.floor(n * trim_pct))
    
    if trim_count >= n // 2:
        trim_count = max(0, n // 2 - 1)
    
    if trim_count > 0:
        trimmed = sorted_vals[trim_count:-trim_count]
    else:
        trimmed = sorted_vals
    
    if len(trimmed) == 0:
        return np.mean(valid)
    
    return float(np.mean(trimmed))


def _winsorized_mean(
    values: np.ndarray,
    winsor_pct: float = 0.1
) -> float:
    """
    Compute Winsorized mean.
    
    Caps extreme values at percentile thresholds instead of removing.
    
    Parameters
    ----------
    values : np.ndarray
        Input values
    winsor_pct : float
        Fraction to Winsorize from each end (default: 0.1 = 10%)
    
    Returns
    -------
    float
        Winsorized mean
    """
    values = np.asarray(values, dtype=np.float64)
    valid = values[~np.isnan(values)]
    
    if len(valid) == 0:
        return np.nan
    
    # Compute percentile thresholds
    lower = np.percentile(valid, winsor_pct * 100)
    upper = np.percentile(valid, (1 - winsor_pct) * 100)
    
    # Cap values
    winsorized = np.clip(valid, lower, upper)
    
    return float(np.mean(winsorized))


def _weighted_mean(
    values: np.ndarray,
    weights: np.ndarray
) -> float:
    """
    Compute weighted mean.
    
    Parameters
    ----------
    values : np.ndarray
        Input values
    weights : np.ndarray
        Weights (same length as values)
    
    Returns
    -------
    float
        Weighted mean
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Handle NaN
    valid_mask = ~np.isnan(values)
    
    if not valid_mask.any():
        return np.nan
    
    valid_values = values[valid_mask]
    valid_weights = weights[valid_mask]
    
    # Normalize weights
    total_weight = np.sum(valid_weights)
    if total_weight <= 0:
        return np.mean(valid_values)
    
    return float(np.sum(valid_values * valid_weights) / total_weight)


def agg_committee(
    list_of_scores: List[float],
    method: str = "trimmed_mean",
    **kwargs
) -> Tuple[float, Dict[str, Any]]:
    """
    Aggregate multiple scores using robust committee method.
    
    Parameters
    ----------
    list_of_scores : List[float]
        List of scores to aggregate (each in [0, 1])
    method : str
        Aggregation method:
        - "trimmed_mean": Remove extremes, then average (default)
        - "winsorized_mean": Cap extremes, then average
        - "median": Simple median
        - "mean": Simple arithmetic mean
        - "weighted": Weighted mean (requires 'weights' kwarg)
    **kwargs
        Method-specific parameters:
        - trim_pct: for trimmed_mean (default: 0.1)
        - winsor_pct: for winsorized_mean (default: 0.1)
        - weights: for weighted method
    
    Returns
    -------
    Tuple[float, Dict[str, Any]]
        - agg_score: Aggregated score in [0, 1]
        - meta: Metadata including method details
    
    Examples
    --------
    >>> scores = [0.7, 0.65, 0.8, 0.3, 0.72]  # One outlier (0.3)
    >>> agg, meta = agg_committee(scores, method="trimmed_mean")
    >>> print(f"Aggregated: {agg:.3f}")  # Less affected by 0.3
    
    Notes
    -----
    Method selection guidance:
    - trimmed_mean: Good default, robust to 1-2 outliers
    - winsorized_mean: Keeps all data points, good for small N
    - median: Most robust but loses nuance
    - mean: Use only when scores are known to be well-behaved
    """
    scores_array = np.asarray(list_of_scores, dtype=np.float64)
    n_scores = len(scores_array)
    
    if n_scores == 0:
        return np.nan, {"method": method, "n_scores": 0, "notes": "Empty input"}
    
    if n_scores == 1:
        return float(scores_array[0]), {
            "method": "single_value",
            "n_scores": 1,
            "notes": "Single input, no aggregation needed"
        }
    
    # Compute aggregation based on method
    if method == "trimmed_mean":
        trim_pct = kwargs.get("trim_pct", 0.1)
        agg_score = _trimmed_mean(scores_array, trim_pct)
        notes = f"Trimmed mean with trim_pct={trim_pct}"
        
    elif method == "winsorized_mean":
        winsor_pct = kwargs.get("winsor_pct", 0.1)
        agg_score = _winsorized_mean(scores_array, winsor_pct)
        notes = f"Winsorized mean with winsor_pct={winsor_pct}"
        
    elif method == "median":
        valid = scores_array[~np.isnan(scores_array)]
        agg_score = float(np.median(valid)) if len(valid) > 0 else np.nan
        notes = "Median aggregation"
        
    elif method == "mean":
        valid = scores_array[~np.isnan(scores_array)]
        agg_score = float(np.mean(valid)) if len(valid) > 0 else np.nan
        notes = "Simple arithmetic mean"
        
    elif method == "weighted":
        weights = kwargs.get("weights")
        if weights is None:
            weights = np.ones(n_scores)
        weights = np.asarray(weights, dtype=np.float64)
        agg_score = _weighted_mean(scores_array, weights)
        notes = f"Weighted mean with weights={weights.tolist()}"
        
    else:
        logger.warning(f"Unknown method '{method}', using trimmed_mean")
        agg_score = _trimmed_mean(scores_array)
        notes = f"Unknown method '{method}', fell back to trimmed_mean"
    
    # Ensure output is in [0, 1]
    agg_score = float(np.clip(agg_score, 0.0, 1.0))
    
    meta = {
        "method": method,
        "n_scores": n_scores,
        "n_valid": int(np.sum(~np.isnan(scores_array))),
        "input_scores": scores_array.tolist(),
        "notes": notes
    }
    
    return agg_score, meta


def committee_series(
    score_series_list: List[np.ndarray],
    method: str = "trimmed_mean",
    **kwargs
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Aggregate multiple score time series element-wise.
    
    Parameters
    ----------
    score_series_list : List[np.ndarray]
        List of score arrays (each T,)
    method : str
        Aggregation method (see agg_committee)
    **kwargs
        Method-specific parameters
    
    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        - agg_t: Aggregated score series
        - meta: Metadata
    
    Examples
    --------
    >>> import numpy as np
    >>> trend = np.random.uniform(0.4, 0.8, 100)
    >>> value = np.random.uniform(0.3, 0.7, 100)
    >>> momentum = np.random.uniform(0.5, 0.9, 100)
    >>> agg_t, meta = committee_series([trend, value, momentum])
    """
    if not score_series_list:
        return np.array([]), {"method": method, "n_series": 0, "notes": "Empty input"}
    
    # Stack into matrix
    n_series = len(score_series_list)
    T = len(score_series_list[0])
    
    # Verify all same length
    for i, s in enumerate(score_series_list):
        if len(s) != T:
            raise ValueError(f"Series {i} has length {len(s)}, expected {T}")
    
    score_matrix = np.column_stack(score_series_list)
    
    # Aggregate row-wise
    agg_t = np.full(T, np.nan)
    
    for t in range(T):
        row_scores = score_matrix[t, :]
        agg_score, _ = agg_committee(row_scores.tolist(), method=method, **kwargs)
        agg_t[t] = agg_score
    
    meta = {
        "method": method,
        "n_series": n_series,
        "T": T,
        "notes": f"Element-wise {method} aggregation of {n_series} series"
    }
    
    return agg_t, meta
