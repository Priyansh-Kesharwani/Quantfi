import numpy as np
from typing import Tuple, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

try:
    from sklearn.covariance import LedoitWolf, ledoit_wolf
    SKLEARN_COV_AVAILABLE = True
except ImportError:
    SKLEARN_COV_AVAILABLE = False
    logger.warning("sklearn.covariance not available. Using simple shrinkage fallback.")


def _simple_shrinkage_covariance(
    returns: np.ndarray,
    shrinkage_target: str = "identity"
) -> Tuple[np.ndarray, float]:
    T, N = returns.shape
    
    sample_cov = np.cov(returns, rowvar=False)
    
    if sample_cov.ndim == 0:                     
        return np.array([[sample_cov]]), 0.0
    
    if shrinkage_target == "identity":
        avg_var = np.trace(sample_cov) / N
        target = avg_var * np.eye(N)
    else:            
        target = np.diag(np.diag(sample_cov))
    
    alpha = min(0.5, N / (T + 1))
    
    shrunk_cov = (1 - alpha) * sample_cov + alpha * target
    
    return shrunk_cov, alpha


def _ledoit_wolf_covariance(returns: np.ndarray) -> Tuple[np.ndarray, float]:
    lw = LedoitWolf()
    lw.fit(returns)
    return lw.covariance_, lw.shrinkage_


def compute_shrinkage_covariance(
    returns: np.ndarray,
    method: str = "auto"
) -> Tuple[np.ndarray, Dict[str, Any]]:
    actual_method = method
    
    if method == "auto":
        actual_method = "ledoit_wolf" if SKLEARN_COV_AVAILABLE else "simple"
    
    if actual_method == "ledoit_wolf":
        if not SKLEARN_COV_AVAILABLE:
            actual_method = "simple"
            logger.warning("Ledoit-Wolf requested but sklearn not available, using simple shrinkage")
    
    if actual_method == "ledoit_wolf":
        cov, shrinkage = _ledoit_wolf_covariance(returns)
        notes = "Ledoit-Wolf optimal shrinkage covariance"
    else:
        cov, shrinkage = _simple_shrinkage_covariance(returns)
        notes = "Simple heuristic shrinkage covariance (sklearn unavailable)"
    
    meta = {
        "method": actual_method,
        "shrinkage_intensity": float(shrinkage),
        "n_assets": cov.shape[0],
        "n_observations": returns.shape[0],
        "notes": notes
    }
    
    return cov, meta


def systemic_coupling(
    cov_matrix: np.ndarray,
    asset_index: int = 0
) -> Tuple[float, Dict[str, Any]]:
    N = cov_matrix.shape[0]
    
    if N == 1:
        return 0.0, {
            "method": "single_asset",
            "n_assets": 1,
            "notes": "Single asset, coupling undefined, returning 0"
        }
    
    std = np.sqrt(np.diag(cov_matrix))
    std = np.maximum(std, 1e-10)                                 
    
    corr_matrix = cov_matrix / np.outer(std, std)
    
    corr_matrix = np.clip(corr_matrix, -1, 1)
    
    other_indices = [j for j in range(N) if j != asset_index]
    
    if not other_indices:
        return 0.0, {"method": "no_other_assets", "n_assets": N, "notes": "No other assets"}
    
    correlations = [corr_matrix[asset_index, j] for j in other_indices]
    avg_abs_corr = np.mean(np.abs(correlations))
    
    chi = float(avg_abs_corr)
    
    meta = {
        "method": "average_abs_correlation",
        "n_assets": N,
        "asset_index": asset_index,
        "avg_correlation": float(np.mean(correlations)),
        "avg_abs_correlation": chi,
        "notes": "Systemic coupling via average absolute correlation"
    }
    
    return chi, meta


def coupling_score(
    asset_returns: np.ndarray,
    market_returns: Union[np.ndarray, None] = None,
    other_assets_returns: Optional[np.ndarray] = None,
    window: int = 63,
    method: str = "auto"
) -> Tuple[np.ndarray, Dict[str, Any]]:
    asset_returns = np.asarray(asset_returns, dtype=np.float64)
    n = len(asset_returns)
    
    if market_returns is not None:
        market_returns = np.asarray(market_returns, dtype=np.float64)
        if other_assets_returns is not None:
            other_assets_returns = np.asarray(other_assets_returns, dtype=np.float64)
            all_returns = np.column_stack([asset_returns, market_returns, other_assets_returns])
        else:
            all_returns = np.column_stack([asset_returns, market_returns])
    elif other_assets_returns is not None:
        other_assets_returns = np.asarray(other_assets_returns, dtype=np.float64)
        all_returns = np.column_stack([asset_returns, other_assets_returns])
    else:
        C_t = np.full(n, 0.5)
        meta = {
            "window_used": window,
            "n_obs": n,
            "method": "neutral_fallback",
            "seed": None,
            "notes": "No market or other asset data — using neutral 0.5"
        }
        return C_t, meta
    
    C_t = np.full(n, np.nan)
    
    for i in range(window, n):
        window_returns = all_returns[i - window:i]
        
        valid_mask = ~np.any(np.isnan(window_returns), axis=1)
        valid_returns = window_returns[valid_mask]
        
        if len(valid_returns) < window // 2:
            continue
        
        cov, cov_meta = compute_shrinkage_covariance(valid_returns, method=method)
        
        chi, chi_meta = systemic_coupling(cov, asset_index=0)
        
        C_t[i] = 1.0 - chi
    
    meta = {
        "window_used": window,
        "n_obs": n,
        "method": method,
        "seed": None,
        "notes": f"Coupling score via {window}-day rolling shrinkage covariance (inverted: high=favorable)"
    }
    
    return C_t, meta


def compute_coupling(asset_returns: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
    return coupling_score(asset_returns, **kwargs)
