"""
Liquidity Indicators

Amihud Illiquidity Ratio measures the price impact of trading.
High illiquidity indicates that trades have large price impact,
which is unfavorable for DCA (higher slippage risk).

For the Gate mechanism:
- High illiquidity → Lower gate value (unfavorable)
- Low illiquidity → Higher gate value (favorable for trading)

References:
-----------
1. Amihud, Y. (2002). "Illiquidity and stock returns: cross-section
   and time-series effects"
2. Pastor, L. & Stambaugh, R.F. (2003). "Liquidity Risk and Expected
   Stock Returns"

Author: Phase 1 Implementation
Date: 2026-02-07
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def amihud_illiquidity(
    return_series: np.ndarray,
    volume_series: np.ndarray,
    window: int = 21
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute Amihud Illiquidity Ratio.
    
    ILLIQ_t = (1/N) × Σ |r_t| / V_t
    
    where r_t is the return and V_t is dollar volume.
    
    Higher ILLIQ → Less liquid → More price impact per unit volume.
    
    Parameters
    ----------
    return_series : np.ndarray
        Return series (can be log returns or simple returns)
    volume_series : np.ndarray
        Volume series (ideally dollar volume, or share volume)
    window : int
        Rolling window for averaging (default: 21 ~ 1 month)
    
    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        - ILL_t: Amihud illiquidity ratio (higher = less liquid)
        - meta: Metadata dict
    
    Examples
    --------
    >>> import numpy as np
    >>> returns = np.random.randn(100) * 0.02
    >>> volumes = np.random.uniform(1e6, 5e6, 100)
    >>> ILL_t, meta = amihud_illiquidity(returns, volumes, window=21)
    >>> print(f"Current illiquidity: {ILL_t[-1]:.2e}")
    
    Notes
    -----
    The Amihud ratio is typically very small (1e-6 to 1e-10 range)
    for liquid assets. It's scale-dependent on volume units.
    
    For cross-asset comparison, normalize using expanding percentile.
    """
    return_series = np.asarray(return_series, dtype=np.float64)
    volume_series = np.asarray(volume_series, dtype=np.float64)
    n = len(return_series)
    
    if len(volume_series) != n:
        raise ValueError("return_series and volume_series must have same length")
    
    # Compute daily illiquidity ratio
    # ILL_daily = |return| / volume
    # Protect against zero volume
    safe_volume = np.maximum(volume_series, 1.0)
    daily_illiq = np.abs(return_series) / safe_volume
    
    # Handle NaN/inf
    daily_illiq = np.where(np.isfinite(daily_illiq), daily_illiq, np.nan)
    
    # Rolling average
    ILL_t = np.full(n, np.nan)
    
    for i in range(window, n + 1):
        window_illiq = daily_illiq[i - window:i]
        valid = window_illiq[~np.isnan(window_illiq)]
        
        if len(valid) >= window // 2:
            ILL_t[i - 1] = np.mean(valid)
    
    meta = {
        "window_used": window,
        "n_obs": n,
        "method": "amihud_ratio",
        "seed": None,
        "notes": f"Amihud illiquidity ratio ({window}-day rolling average)"
    }
    
    return ILL_t, meta


def liquidity_score(
    price_series: np.ndarray,
    volume_series: Optional[np.ndarray] = None,
    window: int = 21,
    pct_lookback: int = 252
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute liquidity score for Gate mechanism.
    
    Returns a score where:
    - High illiquidity (relative to history) → Lower score
    - Low illiquidity → Higher score (favorable)
    
    Parameters
    ----------
    price_series : np.ndarray
        Price series
    volume_series : np.ndarray, optional
        Volume series. If None, returns neutral score with warning.
    window : int
        Window for Amihud calculation
    pct_lookback : int
        Lookback for percentile ranking
    
    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        - L_t: Liquidity score [0, 1] (1 = highly liquid, favorable)
        - meta: Metadata dict
    """
    price_series = np.asarray(price_series, dtype=np.float64)
    n = len(price_series)
    
    # Handle missing volume
    if volume_series is None:
        L_t = np.full(n, 0.5)  # Neutral
        meta = {
            "window_used": window,
            "n_obs": n,
            "method": "neutral_fallback",
            "seed": None,
            "notes": "No volume data. TODO: Connect to volume provider. Using neutral 0.5"
        }
        return L_t, meta
    
    volume_series = np.asarray(volume_series, dtype=np.float64)
    
    # Check volume validity
    valid_vol = volume_series[~np.isnan(volume_series)]
    if len(valid_vol) < n // 4 or np.sum(valid_vol) == 0:
        L_t = np.full(n, 0.5)
        meta = {
            "window_used": window,
            "n_obs": n,
            "method": "neutral_fallback",
            "seed": None,
            "notes": "Insufficient volume data. Using neutral 0.5"
        }
        return L_t, meta
    
    # Compute returns
    safe_prices = np.maximum(price_series, 1e-10)
    returns = np.diff(np.log(safe_prices))
    returns = np.insert(returns, 0, 0)  # Align length
    
    # Compute Amihud illiquidity
    ILL_t, ill_meta = amihud_illiquidity(returns, volume_series, window=window)
    
    # Compute percentile rank (expanding)
    L_t = np.full(n, np.nan)
    
    for i in range(pct_lookback, n):
        current_ill = ILL_t[i]
        if np.isnan(current_ill):
            continue
        
        hist_ill = ILL_t[:i]
        valid_hist = hist_ill[~np.isnan(hist_ill)]
        
        if len(valid_hist) < pct_lookback // 2:
            continue
        
        # Percentile rank (what fraction has HIGHER illiquidity than current)
        # Higher illiquidity = less liquid = lower score
        pct_worse = np.sum(valid_hist >= current_ill) / len(valid_hist)
        L_t[i] = pct_worse  # In [0, 1], higher = more liquid than historical
    
    meta = {
        "window_used": window,
        "n_obs": n,
        "method": "amihud_percentile",
        "seed": None,
        "notes": f"Liquidity score via Amihud percentile (higher = more liquid)",
        "pct_lookback": pct_lookback
    }
    
    return L_t, meta


def compute_liquidity(
    price_series: np.ndarray,
    volume_series: Optional[np.ndarray] = None,
    **kwargs
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Alias for liquidity_score."""
    return liquidity_score(price_series, volume_series, **kwargs)
