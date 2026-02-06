"""
Volatility Indicators

Realized volatility and volatility regime detection for Phase 1 composite.

Volatility serves two roles:
1. Gate component: Extreme volatility may signal market stress (unfavorable)
2. Opportunity component: Higher volatility + drawdown = better entry

The normalized volatility score is used in the Gate mechanism.

Author: Phase 1 Implementation
Date: 2026-02-07
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def realized_vol(
    series: np.ndarray,
    window: int = 21
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute realized volatility as rolling standard deviation of returns.
    
    Parameters
    ----------
    series : np.ndarray
        Price series
    window : int
        Rolling window size (default: 21 ~ 1 month)
    
    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        - vol_t: Array of realized volatility (annualized)
        - meta: Metadata dict
    
    Examples
    --------
    >>> import numpy as np
    >>> prices = np.cumprod(1 + np.random.randn(100) * 0.02)
    >>> vol_t, meta = realized_vol(prices, window=21)
    >>> print(f"Current annualized vol: {vol_t[-1]*100:.1f}%")
    
    Notes
    -----
    Volatility is annualized assuming 252 trading days per year.
    vol_annual = vol_daily * sqrt(252)
    """
    series = np.asarray(series, dtype=np.float64)
    n = len(series)
    
    # Compute log returns
    safe_prices = np.maximum(series, 1e-10)
    log_returns = np.diff(np.log(safe_prices))
    
    # Initialize output (same length as input)
    vol_t = np.full(n, np.nan)
    
    # Rolling standard deviation
    for i in range(window, n):
        ret_window = log_returns[i - window:i]
        valid_rets = ret_window[~np.isnan(ret_window)]
        
        if len(valid_rets) >= window // 2:
            daily_vol = np.std(valid_rets, ddof=1)
            # Annualize
            vol_t[i] = daily_vol * np.sqrt(252)
    
    meta = {
        "window_used": window,
        "n_obs": n,
        "method": "rolling_std",
        "seed": None,
        "notes": f"Annualized realized volatility ({window}-day rolling)"
    }
    
    return vol_t, meta


def volatility_percentile(
    vol_series: np.ndarray,
    lookback: int = 252
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute expanding percentile rank of volatility.
    
    This helps determine if current volatility is high or low
    relative to historical distribution (no lookahead).
    
    Parameters
    ----------
    vol_series : np.ndarray
        Volatility series (from realized_vol)
    lookback : int
        Minimum lookback for percentile calculation
    
    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        - pct_t: Percentile rank [0, 100] (100 = highest volatility)
        - meta: Metadata dict
    
    Notes
    -----
    Uses expanding window: percentile computed on all available history
    up to each point (no lookahead).
    """
    vol_series = np.asarray(vol_series, dtype=np.float64)
    n = len(vol_series)
    
    pct_t = np.full(n, np.nan)
    
    for i in range(lookback, n):
        current_vol = vol_series[i]
        if np.isnan(current_vol):
            continue
        
        # Historical volatility up to (but not including) current
        hist_vol = vol_series[:i]
        valid_hist = hist_vol[~np.isnan(hist_vol)]
        
        if len(valid_hist) < lookback // 2:
            continue
        
        # Percentile rank (what fraction is below current)
        pct_t[i] = 100.0 * np.sum(valid_hist <= current_vol) / len(valid_hist)
    
    meta = {
        "window_used": lookback,
        "n_obs": n,
        "method": "expanding_percentile",
        "seed": None,
        "notes": f"Volatility percentile with {lookback}-observation minimum"
    }
    
    return pct_t, meta


def volatility_regime_score(
    series: np.ndarray,
    vol_window: int = 21,
    pct_lookback: int = 252
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Combined volatility regime indicator.
    
    Returns a score where:
    - High percentile (high vol relative to history) → Lower score
    - Low percentile (calm market) → Higher score
    
    This is used in the Gate mechanism: low vol regime is favorable.
    
    Parameters
    ----------
    series : np.ndarray
        Price series
    vol_window : int
        Window for realized vol calculation
    pct_lookback : int
        Minimum lookback for percentile
    
    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        - V_t: Volatility regime score [0, 1] (1 = calm, favorable)
        - meta: Metadata dict
    """
    # Step 1: Compute realized volatility
    vol_t, vol_meta = realized_vol(series, window=vol_window)
    
    # Step 2: Compute percentile
    pct_t, pct_meta = volatility_percentile(vol_t, lookback=pct_lookback)
    
    # Step 3: Invert percentile to get "calm market" score
    # High percentile (high vol) → low score
    # Low percentile (low vol) → high score
    V_t = (100.0 - pct_t) / 100.0  # Now in [0, 1]
    
    meta = {
        "window_used": vol_window,
        "n_obs": len(series),
        "method": "volatility_regime",
        "seed": None,
        "notes": f"Volatility regime score: vol_window={vol_window}, pct_lookback={pct_lookback}",
        "sub_meta": {
            "realized_vol": vol_meta,
            "percentile": pct_meta
        }
    }
    
    return V_t, meta


# Convenience aliases
def compute_volatility(series: np.ndarray, window: int = 21) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Alias for realized_vol."""
    return realized_vol(series, window)


def vol_percentile_score(series: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Alias for volatility_regime_score."""
    return volatility_regime_score(series, **kwargs)
