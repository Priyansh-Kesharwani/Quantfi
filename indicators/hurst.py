"""
Hurst Exponent Estimator

The Hurst exponent H ∈ (0, 1) measures the persistence or mean-reversion
tendency of a time series:
- H > 0.5: Persistent (trending) - momentum strategies favored
- H = 0.5: Random walk - no predictable pattern
- H < 0.5: Mean-reverting - contrarian strategies favored

This module provides a wavelet-based Hurst estimator wrapper.

References:
-----------
1. Hurst, H.E. (1951). "Long-term storage capacity of reservoirs"
2. Flandrin, P. (1992). "Wavelet analysis and synthesis of fractional Brownian motion"
3. Abry, P. & Veitch, D. (1998). "Wavelet analysis of long-range dependent traffic"

Implementation Notes:
--------------------
- Primary method: Wavelet-based estimator using pywt (if available)
- Fallback: Rescaled Range (R/S) method - deterministic surrogate
- The R/S method is less accurate but always available

Author: Phase 1 Implementation
Date: 2026-02-07
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import PyWavelets for wavelet-based estimation
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    logger.warning("pywt not available. Using R/S fallback for Hurst estimation.")


def _rescaled_range_hurst(series: np.ndarray, max_k: Optional[int] = None) -> float:
    """
    Rescaled Range (R/S) method for Hurst exponent estimation.
    
    This is a deterministic fallback when wavelet methods are unavailable.
    
    Parameters
    ----------
    series : np.ndarray
        1D array of prices or returns
    max_k : int, optional
        Maximum number of divisions for R/S analysis
    
    Returns
    -------
    float
        Estimated Hurst exponent
    
    Notes
    -----
    The R/S statistic is computed as:
    R/S = (max(cumsum(x - mean)) - min(cumsum(x - mean))) / std(x)
    
    For a series with Hurst exponent H:
    E[R/S] ∝ n^H
    
    We estimate H by linear regression of log(R/S) vs log(n).
    """
    n = len(series)
    if n < 20:
        return 0.5  # Insufficient data, return random walk assumption
    
    if max_k is None:
        max_k = int(np.log2(n)) - 1
    
    max_k = max(2, min(max_k, int(np.log2(n)) - 1))
    
    rs_values = []
    n_values = []
    
    for k in range(max_k):
        # Divide series into 2^k subseries
        n_sub = n // (2 ** k)
        if n_sub < 8:
            break
        
        rs_list = []
        for i in range(2 ** k):
            start = i * n_sub
            end = start + n_sub
            subseries = series[start:end]
            
            if len(subseries) < 8:
                continue
            
            # Compute R/S for this subseries
            mean_sub = np.mean(subseries)
            std_sub = np.std(subseries, ddof=1)
            
            if std_sub < 1e-10:
                continue
            
            # Cumulative deviation from mean
            cumdev = np.cumsum(subseries - mean_sub)
            
            # Rescaled range
            r = np.max(cumdev) - np.min(cumdev)
            rs = r / std_sub
            rs_list.append(rs)
        
        if rs_list:
            rs_values.append(np.mean(rs_list))
            n_values.append(n_sub)
    
    if len(rs_values) < 2:
        return 0.5  # Insufficient data
    
    # Linear regression: log(R/S) = H * log(n) + c
    log_n = np.log(n_values)
    log_rs = np.log(rs_values)
    
    # Simple least squares
    n_points = len(log_n)
    sum_x = np.sum(log_n)
    sum_y = np.sum(log_rs)
    sum_xy = np.sum(log_n * log_rs)
    sum_x2 = np.sum(log_n ** 2)
    
    denom = n_points * sum_x2 - sum_x ** 2
    if abs(denom) < 1e-10:
        return 0.5
    
    H = (n_points * sum_xy - sum_x * sum_y) / denom
    
    # Clamp to valid range
    return float(np.clip(H, 0.01, 0.99))


def _wavelet_hurst(series: np.ndarray, wavelet: str = 'db4', 
                   max_level: Optional[int] = None) -> float:
    """
    Wavelet-based Hurst exponent estimation.
    
    Uses the variance of wavelet detail coefficients at different scales
    to estimate the Hurst exponent.
    
    Parameters
    ----------
    series : np.ndarray
        1D array of prices or returns
    wavelet : str
        Wavelet to use (default: 'db4' - Daubechies 4)
    max_level : int, optional
        Maximum decomposition level
    
    Returns
    -------
    float
        Estimated Hurst exponent
    
    Notes
    -----
    For a fractional Brownian motion with Hurst exponent H:
    Var(d_j) ∝ 2^(j(2H-1))
    
    where d_j are the detail coefficients at scale j.
    
    We estimate H by linear regression of log(Var(d_j)) vs j.
    """
    if not PYWT_AVAILABLE:
        raise ImportError("pywt required for wavelet Hurst estimation")
    
    n = len(series)
    if n < 32:
        return 0.5  # Insufficient data
    
    if max_level is None:
        max_level = int(np.log2(n)) - 2
    
    max_level = max(2, min(max_level, pywt.dwt_max_level(n, wavelet)))
    
    # Wavelet decomposition
    coeffs = pywt.wavedec(series, wavelet, level=max_level)
    
    # Compute variance at each scale (skip approximation coefficients)
    variances = []
    scales = []
    
    for j in range(1, len(coeffs)):
        d_j = coeffs[j]
        if len(d_j) < 4:
            continue
        var_j = np.var(d_j)
        if var_j > 1e-15:
            variances.append(var_j)
            scales.append(j)
    
    if len(variances) < 2:
        return 0.5  # Insufficient data
    
    # Linear regression: log(Var) = (2H-1) * j + c
    scales = np.array(scales)
    log_var = np.log(variances)
    
    # Simple least squares
    n_points = len(scales)
    sum_x = np.sum(scales)
    sum_y = np.sum(log_var)
    sum_xy = np.sum(scales * log_var)
    sum_x2 = np.sum(scales ** 2)
    
    denom = n_points * sum_x2 - sum_x ** 2
    if abs(denom) < 1e-10:
        return 0.5
    
    slope = (n_points * sum_xy - sum_x * sum_y) / denom
    
    # H = (slope + 1) / 2
    # Note: slope should be negative for typical financial series
    H = (slope / np.log(2) + 1) / 2
    
    # Clamp to valid range
    return float(np.clip(H, 0.01, 0.99))


def estimate_hurst(
    series: np.ndarray,
    window: Optional[int] = None,
    method: str = "auto"
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Estimate Hurst exponent for a time series.
    
    Returns a time series of Ĥ_t values computed on rolling windows.
    
    Parameters
    ----------
    series : np.ndarray
        1D array of prices (not returns). The function will compute
        log returns internally.
    window : int, optional
        Rolling window size. Default: 252 (1 trading year)
    method : str
        Estimation method:
        - "auto": Use wavelet if pywt available, else R/S
        - "wavelet": Force wavelet method (requires pywt)
        - "rs": Force Rescaled Range method
    
    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        - H_t: Array of Hurst estimates (same length as input, NaN for warmup)
        - meta: Metadata dict with keys:
            - window_used: int
            - n_obs: int
            - method: str
            - seed: None (deterministic)
            - notes: str (any warnings or fallback notes)
    
    Examples
    --------
    >>> import numpy as np
    >>> prices = np.cumprod(1 + np.random.randn(500) * 0.01)
    >>> H_t, meta = estimate_hurst(prices, window=252)
    >>> print(f"Latest H: {H_t[-1]:.3f}, Method: {meta['method']}")
    
    Notes
    -----
    The Hurst exponent is estimated on log returns, not prices.
    This removes the trend component and focuses on the scaling behavior.
    
    References
    ----------
    - Flandrin (1992): Wavelet analysis of fBm
    - Peters (1994): Fractal Market Analysis
    """
    series = np.asarray(series, dtype=np.float64)
    n = len(series)
    
    if window is None:
        window = min(252, n // 2)
    
    # Choose method
    actual_method = method
    notes = ""
    
    if method == "auto":
        if PYWT_AVAILABLE:
            actual_method = "wavelet"
        else:
            actual_method = "rs"
            notes = "pywt not available, using R/S fallback"
    elif method == "wavelet" and not PYWT_AVAILABLE:
        actual_method = "rs"
        notes = "wavelet requested but pywt not available, using R/S fallback"
    
    # Compute log returns
    # Protect against zero/negative prices
    safe_prices = np.maximum(series, 1e-10)
    log_returns = np.diff(np.log(safe_prices))
    
    # Initialize output
    H_t = np.full(n, np.nan)
    
    # Rolling estimation
    for i in range(window, n):
        # Window of log returns
        ret_window = log_returns[i - window:i]
        
        if len(ret_window) < window:
            continue
        
        # Estimate Hurst
        if actual_method == "wavelet":
            try:
                H_t[i] = _wavelet_hurst(ret_window)
            except Exception as e:
                logger.warning(f"Wavelet estimation failed at index {i}: {e}")
                H_t[i] = _rescaled_range_hurst(ret_window)
        else:
            H_t[i] = _rescaled_range_hurst(ret_window)
    
    meta = {
        "window_used": window,
        "n_obs": n,
        "method": actual_method,
        "seed": None,  # Deterministic
        "notes": notes if notes else "Estimation completed successfully"
    }
    
    return H_t, meta


# Alias for backward compatibility
hurst_exponent = estimate_hurst
