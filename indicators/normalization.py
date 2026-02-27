import numpy as np
import pandas as pd
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

def expanding_ecdf_sigmoid(
    series: pd.Series,
    k: float = 1.0,
    polarity: int = 1,
    min_obs: int = 100
) -> pd.Series:
    """Full expanding-ECDF → inverse-normal → sigmoid pipeline.

    Steps (per the Phase A spec):
      1. Expanding ECDF  →  p_t ∈ (0, 1)
      2. Inverse-normal  →  z_t = Φ⁻¹(p_t)
      3. Sigmoid          →  s_t = 1 / (1 + e^{-k·z_t})
      4. Polarity align   →  flip if polarity == -1

    Parameters
    ----------
    series : pd.Series or array-like
        Raw indicator values (UTC-indexed preferred).
    k : float
        Sigmoid steepness parameter.
    polarity : int
        +1 keeps the score direction; -1 inverts (lower raw → higher score).
    min_obs : int
        Minimum observations before first valid output.

    Returns
    -------
    pd.Series
        Normalized scores in (0, 1), NaN during warm-up.
    """
    if isinstance(series, pd.Series):
        index = series.index
        values = series.values.astype(np.float64)
    else:
        values = np.asarray(series, dtype=np.float64)
        index = None

    higher_is_favorable = (polarity >= 0)
    score_arr, _meta = normalize_to_score(
        values, min_obs=min_obs, k=k, higher_is_favorable=higher_is_favorable
    )

    if index is not None:
        return pd.Series(score_arr, index=index, name="ecdf_sigmoid")
    return pd.Series(score_arr, name="ecdf_sigmoid")

def _expanding_midrank_ecdf(raw: np.ndarray, min_obs: int) -> np.ndarray:
    """Expanding ECDF with exact midrank tie rule.

    p_t = (rank_less_t + 0.5 * rank_equal_t) / n_t
    """
    raw = np.asarray(raw, dtype=np.float64)
    n = len(raw)
    pct_t = np.full(n, np.nan)
    for t in range(min_obs, n):
        current = raw[t]
        if np.isnan(current):
            continue
        hist = raw[: t + 1]
        valid = hist[~np.isnan(hist)]
        if len(valid) < min_obs:
            continue
        n_t = len(valid)
        rank_less = np.sum(valid < current)
        rank_equal = np.sum(valid == current)
        pct_t[t] = (rank_less + 0.5 * rank_equal) / n_t
    return pct_t

def canonical_normalize(
    raw: np.ndarray,
    k: float = 1.0,
    eps: float = 1e-9,
    mode: str = "exact",
    higher_is_favorable: bool = True,
    min_obs: int = 1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Expand ECDF -> inverse-normal -> sigmoid with configurable polarity.

    Parameters
    ----------
    raw : 1d array of raw values.
    k : sigmoid steepness.
    eps : clip probability to [eps, 1-eps] before Phi^{-1}.
    mode : "exact" (midrank tie rule) or "approx" (expanding_percentile).
    higher_is_favorable : if False, s_t <- 1 - s_t.
    min_obs : minimum observations before producing non-NaN.

    Returns
    -------
    s_t : scores in (0, 1).
    meta : dict with k, eps, mode, higher_is_favorable, n_obs, method.
    """
    raw = np.asarray(raw, dtype=np.float64)

    if mode == "exact":
        pct_t = _expanding_midrank_ecdf(raw, min_obs)
        pct_t = np.clip(pct_t, eps, 1.0 - eps)
        z_t = stats.norm.ppf(pct_t)
        z_t = np.where(np.isnan(pct_t), np.nan, z_t)
        safe_z = np.clip(z_t * k, -500, 500)
        s_t = 1.0 / (1.0 + np.exp(-safe_z))
        s_t = np.where(np.isnan(z_t), np.nan, s_t)
        eps_f = np.finfo(np.float64).eps
        s_t = np.clip(s_t, eps_f, 1.0 - eps_f)
    elif mode == "approx":
        pct_t, _ = expanding_percentile(raw, min_obs=min_obs)
        pct_t = np.clip(pct_t, eps, 1.0 - eps)
        z_t = percentile_to_z(pct_t)
        s_t = z_to_sigmoid(z_t, k=k)
    else:
        raise NotImplementedError(
            f"canonical_normalize mode={mode!r} not implemented; use 'exact' or 'approx'."
        )

    if not higher_is_favorable:
        s_t = polarity_align(s_t, higher_is_favorable=False)

    meta: Dict[str, Any] = {
        "k": k,
        "eps": eps,
        "mode": mode,
        "higher_is_favorable": higher_is_favorable,
        "n_obs": int(np.sum(~np.isnan(s_t))),
        "method": "canonical_normalize",
    }
    return s_t, meta
