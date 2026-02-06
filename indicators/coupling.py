"""
Systemic Coupling Indicator

Measures the degree of correlation/coupling between an asset and the
broader market or other assets. High systemic coupling indicates that
the asset moves with the market, which may be unfavorable during
market-wide stress events.

For the Gate mechanism:
- High coupling → Asset moves with market → More exposed to systemic risk
- Low coupling → More idiosyncratic → Less systemic risk

Implementation:
--------------
We use Ledoit-Wolf shrinkage covariance estimation for robustness.
Shrinkage estimators are more stable than sample covariance, especially
with limited data or during regime changes.

References:
-----------
1. Ledoit, O. & Wolf, M. (2004). "A well-conditioned estimator for
   large-dimensional covariance matrices"
2. Chen, Y. et al. (2010). "Shrinkage algorithms for MMSE covariance
   estimation"

Author: Phase 1 Implementation
Date: 2026-02-07
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

# Try to import sklearn for Ledoit-Wolf
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
    """
    Simple shrinkage covariance estimator (fallback).
    
    Cov_shrunk = (1-α) × Cov_sample + α × Target
    
    where α is the shrinkage intensity (heuristically chosen).
    
    Parameters
    ----------
    returns : np.ndarray
        Return matrix (T × N) where T=observations, N=assets
    shrinkage_target : str
        Target matrix type: "identity" or "diagonal"
    
    Returns
    -------
    Tuple[np.ndarray, float]
        - Shrunk covariance matrix (N × N)
        - Shrinkage intensity used
    """
    T, N = returns.shape
    
    # Sample covariance
    sample_cov = np.cov(returns, rowvar=False)
    
    if sample_cov.ndim == 0:  # Single asset case
        return np.array([[sample_cov]]), 0.0
    
    # Shrinkage target
    if shrinkage_target == "identity":
        # Scale identity to match sample cov trace
        avg_var = np.trace(sample_cov) / N
        target = avg_var * np.eye(N)
    else:  # diagonal
        target = np.diag(np.diag(sample_cov))
    
    # Heuristic shrinkage intensity
    # More shrinkage when T/N is small
    alpha = min(0.5, N / (T + 1))
    
    # Shrunk estimate
    shrunk_cov = (1 - alpha) * sample_cov + alpha * target
    
    return shrunk_cov, alpha


def _ledoit_wolf_covariance(returns: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Ledoit-Wolf shrinkage covariance estimator.
    
    Automatically determines optimal shrinkage intensity.
    """
    lw = LedoitWolf()
    lw.fit(returns)
    return lw.covariance_, lw.shrinkage_


def compute_shrinkage_covariance(
    returns: np.ndarray,
    method: str = "auto"
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute shrinkage covariance matrix.
    
    Parameters
    ----------
    returns : np.ndarray
        Return matrix (T × N)
    method : str
        "auto" (use LW if available), "ledoit_wolf", or "simple"
    
    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        - Covariance matrix
        - Meta with shrinkage intensity and method
    """
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
    """
    Compute systemic coupling (χ_t) from covariance matrix.
    
    Systemic coupling measures how correlated an asset is with others.
    
    χ = (1/(N-1)) × Σ_{j≠i} |corr(i,j)|
    
    Parameters
    ----------
    cov_matrix : np.ndarray
        Covariance matrix (N × N)
    asset_index : int
        Index of the asset to compute coupling for
    
    Returns
    -------
    Tuple[float, Dict[str, Any]]
        - χ: Coupling score [0, 1] where 1 = perfectly correlated with all
        - meta: Metadata
    
    Notes
    -----
    Why use shrinkage covariance?
    - Sample covariance is noisy with limited data
    - Shrinkage provides a regularized, more stable estimate
    - Better behaved eigenvalues (no near-zero or negative)
    - More robust during regime changes
    """
    N = cov_matrix.shape[0]
    
    if N == 1:
        # Single asset, no coupling
        return 0.0, {
            "method": "single_asset",
            "n_assets": 1,
            "notes": "Single asset, coupling undefined, returning 0"
        }
    
    # Convert to correlation matrix
    std = np.sqrt(np.diag(cov_matrix))
    std = np.maximum(std, 1e-10)  # Protect against zero variance
    
    corr_matrix = cov_matrix / np.outer(std, std)
    
    # Clip to valid correlation range
    corr_matrix = np.clip(corr_matrix, -1, 1)
    
    # Average absolute correlation with other assets
    other_indices = [j for j in range(N) if j != asset_index]
    
    if not other_indices:
        return 0.0, {"method": "no_other_assets", "n_assets": N, "notes": "No other assets"}
    
    correlations = [corr_matrix[asset_index, j] for j in other_indices]
    avg_abs_corr = np.mean(np.abs(correlations))
    
    # Chi is in [0, 1]
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
    """
    Compute rolling systemic coupling score.
    
    For the Gate mechanism, we want:
    - High coupling (moves with market) → Lower gate value
    - Low coupling → Higher gate value (more independent)
    
    Parameters
    ----------
    asset_returns : np.ndarray
        Returns of the target asset (T,)
    market_returns : np.ndarray, optional
        Market/benchmark returns (T,). If None and no other_assets,
        returns neutral score.
    other_assets_returns : np.ndarray, optional
        Returns of other assets (T × M)
    window : int
        Rolling window for covariance estimation
    method : str
        Covariance estimation method
    
    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        - C_t: Coupling score [0, 1] (1 = low coupling, favorable)
        - meta: Metadata dict
    
    Notes
    -----
    Score is inverted: low raw coupling → high score (favorable for DCA).
    """
    asset_returns = np.asarray(asset_returns, dtype=np.float64)
    n = len(asset_returns)
    
    # Build return matrix
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
        # No comparison data, return neutral
        C_t = np.full(n, 0.5)
        meta = {
            "window_used": window,
            "n_obs": n,
            "method": "neutral_fallback",
            "seed": None,
            "notes": "No market or other asset data. TODO: Connect to market data provider. Using neutral 0.5"
        }
        return C_t, meta
    
    # Rolling coupling computation
    C_t = np.full(n, np.nan)
    
    for i in range(window, n):
        window_returns = all_returns[i - window:i]
        
        # Remove rows with NaN
        valid_mask = ~np.any(np.isnan(window_returns), axis=1)
        valid_returns = window_returns[valid_mask]
        
        if len(valid_returns) < window // 2:
            continue
        
        # Compute shrinkage covariance
        cov, cov_meta = compute_shrinkage_covariance(valid_returns, method=method)
        
        # Compute coupling (asset is at index 0)
        chi, chi_meta = systemic_coupling(cov, asset_index=0)
        
        # Invert: low coupling → high score (favorable)
        C_t[i] = 1.0 - chi
    
    meta = {
        "window_used": window,
        "n_obs": n,
        "method": method,
        "seed": None,
        "notes": f"Coupling score via {window}-day rolling shrinkage covariance (inverted: high=favorable)"
    }
    
    return C_t, meta


# Convenience alias
def compute_coupling(asset_returns: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Alias for coupling_score."""
    return coupling_score(asset_returns, **kwargs)
