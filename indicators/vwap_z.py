"""
VWAP-based Valuation Z-Score

Volume-Weighted Average Price (VWAP) provides a fair value benchmark.
The Z-score measures deviation from this benchmark:
- Negative Z → Price below VWAP → Potentially undervalued
- Positive Z → Price above VWAP → Potentially overvalued

For DCA modulation:
- Negative Z_vwap is favorable (buying below fair value)
- We invert polarity so higher normalized score = more favorable

Author: Phase 1 Implementation
Date: 2026-02-07
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def _compute_rolling_vwap(
    price: np.ndarray,
    volume: np.ndarray,
    window: int
) -> np.ndarray:
    """
    Compute rolling VWAP.
    
    VWAP = Σ(Price × Volume) / Σ(Volume)
    
    Parameters
    ----------
    price : np.ndarray
        Price series
    volume : np.ndarray
        Volume series
    window : int
        Rolling window size
    
    Returns
    -------
    np.ndarray
        Rolling VWAP series
    """
    n = len(price)
    vwap = np.full(n, np.nan)
    
    for i in range(window, n + 1):
        p = price[i - window:i]
        v = volume[i - window:i]
        
        # Handle missing values
        mask = ~(np.isnan(p) | np.isnan(v))
        if not mask.any():
            continue
        
        p_valid = p[mask]
        v_valid = v[mask]
        
        total_vol = np.sum(v_valid)
        if total_vol > 0:
            vwap[i - 1] = np.sum(p_valid * v_valid) / total_vol
    
    return vwap


def _compute_rolling_sma(price: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling Simple Moving Average (SMA) as VWAP fallback.
    
    When volume data is unavailable, we use SMA as a proxy for fair value.
    """
    n = len(price)
    sma = np.full(n, np.nan)
    
    for i in range(window, n + 1):
        p = price[i - window:i]
        valid = p[~np.isnan(p)]
        if len(valid) > 0:
            sma[i - 1] = np.mean(valid)
    
    return sma


def _compute_z_score(
    price: np.ndarray,
    benchmark: np.ndarray,
    vol_window: int
) -> np.ndarray:
    """
    Compute Z-score of price relative to benchmark.
    
    Z = (Price - Benchmark) / Rolling_Std(Price)
    
    Parameters
    ----------
    price : np.ndarray
        Current price series
    benchmark : np.ndarray
        VWAP or SMA benchmark
    vol_window : int
        Window for volatility estimation
    
    Returns
    -------
    np.ndarray
        Z-score series
    """
    n = len(price)
    z_score = np.full(n, np.nan)
    
    for i in range(vol_window, n):
        if np.isnan(benchmark[i]) or np.isnan(price[i]):
            continue
        
        # Rolling std of price
        p_window = price[max(0, i - vol_window):i + 1]
        valid_prices = p_window[~np.isnan(p_window)]
        
        if len(valid_prices) < vol_window // 2:
            continue
        
        std = np.std(valid_prices, ddof=1)
        if std < 1e-10:
            continue
        
        z_score[i] = (price[i] - benchmark[i]) / std
    
    return z_score


def compute_vwap_z(
    price_series: np.ndarray,
    volume_series: Optional[np.ndarray] = None,
    window: int = 20
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute VWAP-based Z-score for valuation assessment.
    
    Z_vwap < 0 indicates price is below VWAP (potentially undervalued).
    For DCA, negative Z is favorable - we buy below fair value.
    
    Parameters
    ----------
    price_series : np.ndarray
        Price series (Close prices)
    volume_series : np.ndarray, optional
        Volume series. If None, falls back to MA-based Z.
    window : int
        Rolling window for VWAP/MA calculation (default: 20)
    
    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        - Z_t: Array of Z-scores (negative = undervalued)
        - meta: Metadata dict with keys:
            - window_used: int
            - n_obs: int
            - method: str ("vwap" or "sma_fallback")
            - seed: None
            - notes: str
    
    Examples
    --------
    >>> import numpy as np
    >>> prices = np.array([100, 102, 98, 95, 97, 100, 103, 101, 99, 98])
    >>> volumes = np.array([1000, 1200, 800, 1500, 900, 1100, 1300, 1000, 1400, 1200])
    >>> Z_t, meta = compute_vwap_z(prices, volumes, window=5)
    >>> print(f"Latest Z: {Z_t[-1]:.2f}")
    
    Notes
    -----
    Interpretation:
    - Z < -2: Extreme undervaluation
    - Z < -1: Moderate undervaluation
    - -1 < Z < 1: Fair value range
    - Z > 1: Overvalued
    - Z > 2: Extreme overvaluation
    
    For DCA scoring, we invert polarity: more negative Z → higher score.
    """
    price_series = np.asarray(price_series, dtype=np.float64)
    n = len(price_series)
    
    notes = ""
    
    # Check if volume is available and valid
    if volume_series is not None:
        volume_series = np.asarray(volume_series, dtype=np.float64)
        
        # Check for valid volume data
        valid_volume = volume_series[~np.isnan(volume_series)]
        if len(valid_volume) > n // 2 and np.sum(valid_volume) > 0:
            # Use VWAP
            benchmark = _compute_rolling_vwap(price_series, volume_series, window)
            method = "vwap"
        else:
            # Fallback to SMA
            benchmark = _compute_rolling_sma(price_series, window)
            method = "sma_fallback"
            notes = "Volume data insufficient, using SMA as VWAP proxy"
            # TODO: Implement proper VWAP data provider for production
    else:
        # No volume provided, use SMA fallback
        benchmark = _compute_rolling_sma(price_series, window)
        method = "sma_fallback"
        notes = "No volume data provided. TODO: Connect to volume data provider"
    
    # Compute Z-score
    vol_window = window  # Use same window for volatility
    Z_t = _compute_z_score(price_series, benchmark, vol_window)
    
    meta = {
        "window_used": window,
        "n_obs": n,
        "method": method,
        "seed": None,  # Deterministic
        "notes": notes if notes else f"VWAP Z-score computed with {window}-day window"
    }
    
    return Z_t, meta


def vwap_undervaluation_score(
    price_series: np.ndarray,
    volume_series: Optional[np.ndarray] = None,
    window: int = 20
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convenience wrapper that returns undervaluation score (inverted Z).
    
    Higher score = more undervalued = more favorable for DCA.
    """
    Z_t, meta = compute_vwap_z(price_series, volume_series, window)
    
    # Invert: negative Z (undervalued) becomes positive score
    # We'll handle normalization in the normalization module
    U_t = -Z_t
    
    meta["notes"] = meta.get("notes", "") + " | Polarity inverted for undervaluation score"
    
    return U_t, meta
