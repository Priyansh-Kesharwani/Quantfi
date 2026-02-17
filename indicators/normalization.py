import numpy as np
from typing import Tuple, Dict, Any, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)


def expanding_percentile(
    series: np.ndarray,
    min_obs: int = 100
) -> Tuple[np.ndarray, Dict[str, Any]]:
    series = np.asarray(series, dtype=np.float64)
    n = len(series)
    
    pct_t = np.full(n, np.nan)
    sample_sizes = np.full(n, 0, dtype=int)
    
    for t in range(min_obs, n):
        current_value = series[t]
        if np.isnan(current_value):
            continue
        
        historical = series[:t]
        valid_hist = historical[~np.isnan(historical)]
        
        if len(valid_hist) < min_obs:
            continue
        
        sample_sizes[t] = len(valid_hist)
        
        pct = np.sum(valid_hist <= current_value) / len(valid_hist)
        
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
    percentile = np.asarray(percentile, dtype=np.float64)
    
    safe_pct = np.clip(percentile, 0.001, 0.999)
    
    z = stats.norm.ppf(safe_pct)
    
    z = np.where(np.isnan(percentile), np.nan, z)
    
    return z


def z_to_sigmoid(
    z: np.ndarray,
    k: float = 1.0
) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    
    safe_z = np.clip(z * k, -500, 500)
    
    sigmoid = 1.0 / (1.0 + np.exp(-safe_z))
    eps = np.finfo(np.float64).eps
    sigmoid = np.clip(sigmoid, eps, 1.0 - eps)
    
    sigmoid = np.where(np.isnan(z), np.nan, sigmoid)
    
    return sigmoid


def polarity_align(
    score: np.ndarray,
    higher_is_favorable: bool = True
) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64)
    
    if higher_is_favorable:
        return score
    else:
        return np.round(1.0 - score, 15)


def normalize_to_score(
    raw_values: np.ndarray,
    min_obs: int = 100,
    k: float = 1.0,
    higher_is_favorable: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    pct_t, pct_meta = expanding_percentile(raw_values, min_obs=min_obs)
    
    z_t = percentile_to_z(pct_t)
    
    sigmoid_t = z_to_sigmoid(z_t, k=k)
    
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
