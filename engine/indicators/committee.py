import numpy as np
from typing import Tuple, Dict, Any, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


def _trimmed_mean(
    values: np.ndarray,
    trim_pct: float = 0.1
) -> float:
    values = np.asarray(values, dtype=np.float64)
    valid = values[~np.isnan(values)]
    
    if len(valid) == 0:
        return np.nan
    
    if len(valid) <= 2:
        return np.mean(valid)
    
    sorted_vals = np.sort(valid)
    n = len(sorted_vals)
    
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
    values = np.asarray(values, dtype=np.float64)
    valid = values[~np.isnan(values)]
    
    if len(valid) == 0:
        return np.nan
    
    lower = np.percentile(valid, winsor_pct * 100)
    upper = np.percentile(valid, (1 - winsor_pct) * 100)
    
    winsorized = np.clip(valid, lower, upper)
    
    return float(np.mean(winsorized))


def _weighted_mean(
    values: np.ndarray,
    weights: np.ndarray
) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    valid_mask = ~np.isnan(values)
    
    if not valid_mask.any():
        return np.nan
    
    valid_values = values[valid_mask]
    valid_weights = weights[valid_mask]
    
    total_weight = np.sum(valid_weights)
    if total_weight <= 0:
        return np.mean(valid_values)
    
    return float(np.sum(valid_values * valid_weights) / total_weight)


def agg_committee(
    list_of_scores: List[float],
    method: str = "trimmed_mean",
    **kwargs
) -> Tuple[float, Dict[str, Any]]:
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
    if not score_series_list:
        return np.array([]), {"method": method, "n_series": 0, "notes": "Empty input"}
    
    n_series = len(score_series_list)
    T = len(score_series_list[0])
    
    for i, s in enumerate(score_series_list):
        if len(s) != T:
            raise ValueError(f"Series {i} has length {len(s)}, expected {T}")
    
    score_matrix = np.column_stack(score_series_list)
    
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
