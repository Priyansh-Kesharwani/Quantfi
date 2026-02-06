"""
Undervaluation Indicator (U_t)

Measures statistical undervaluation based on price deviation from fair value.
Used in the Opportunity block of the composite score.

Methods:
- Z-score vs VWAP (Volume-Weighted Average Price)
- Z-score vs long-term moving average
- Drawdown from rolling maximum

The undervaluation signal is weighted by persistence (Hurst) in the composite:
U_weighted = U_t × g_pers(H_t)

This ensures we only buy dips in persistent/trending regimes, not in 
mean-reverting or choppy markets.

Author: Phase 2 Implementation
Date: 2026-02-07
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def vwap_approximation(
    prices: np.ndarray,
    volumes: np.ndarray,
    window: int = 20
) -> np.ndarray:
    """
    Approximate VWAP from daily OHLCV data.
    
    True VWAP requires intraday data. For daily data, we approximate:
    VWAP ≈ Rolling sum(TypicalPrice × Volume) / Rolling sum(Volume)
    
    where TypicalPrice = (High + Low + Close) / 3
    For simplicity, if no high/low, use Close.
    """
    prices = np.asarray(prices, dtype=np.float64)
    volumes = np.asarray(volumes, dtype=np.float64)
    n = len(prices)
    
    # Handle zero volumes
    volumes = np.maximum(volumes, 1.0)
    
    pv = prices * volumes
    
    vwap = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        sum_pv = np.sum(pv[i - window + 1:i + 1])
        sum_v = np.sum(volumes[i - window + 1:i + 1])
        vwap[i] = sum_pv / max(sum_v, 1.0)
    
    return vwap


def price_vwap_zscore(
    prices: np.ndarray,
    volumes: Optional[np.ndarray] = None,
    vwap_window: int = 20,
    zscore_window: int = 50
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute Z-score of price relative to VWAP.
    
    Z = (Price - VWAP) / std(Price - VWAP)
    
    Negative Z → Price below VWAP → Potentially undervalued
    Positive Z → Price above VWAP → Potentially overvalued
    
    Parameters
    ----------
    prices : np.ndarray
        Close prices
    volumes : np.ndarray, optional
        Volume data. If None, uses simple moving average instead of VWAP.
    vwap_window : int
        Window for VWAP calculation (default: 20)
    zscore_window : int
        Window for Z-score rolling stats (default: 50)
    
    Returns
    -------
    Tuple[np.ndarray, Dict]
        - z_t: Z-score series (negative = undervalued)
        - meta: Metadata
    """
    prices = np.asarray(prices, dtype=np.float64)
    n = len(prices)
    
    # Compute anchor (VWAP or SMA)
    if volumes is not None and np.sum(volumes) > 0:
        volumes = np.asarray(volumes, dtype=np.float64)
        anchor = vwap_approximation(prices, volumes, vwap_window)
        anchor_method = "vwap"
    else:
        # Fallback to simple moving average
        anchor = np.full(n, np.nan)
        for i in range(vwap_window - 1, n):
            anchor[i] = np.mean(prices[i - vwap_window + 1:i + 1])
        anchor_method = "sma"
    
    # Price deviation from anchor
    deviation = prices - anchor
    
    # Rolling Z-score
    z_t = np.full(n, np.nan)
    
    for i in range(max(vwap_window, zscore_window) - 1, n):
        dev_window = deviation[i - zscore_window + 1:i + 1]
        valid_dev = dev_window[~np.isnan(dev_window)]
        
        if len(valid_dev) < zscore_window // 2:
            continue
        
        mean_dev = np.mean(valid_dev)
        std_dev = np.std(valid_dev, ddof=1)
        
        if std_dev > 1e-10:
            z_t[i] = (deviation[i] - mean_dev) / std_dev
    
    meta = {
        "vwap_window": vwap_window,
        "zscore_window": zscore_window,
        "anchor_method": anchor_method,
        "method": "price_vwap_zscore",
        "notes": "Z-score vs VWAP/SMA: negative = undervalued, positive = overvalued"
    }
    
    return z_t, meta


def drawdown_score(
    prices: np.ndarray,
    lookback: int = 252
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Compute drawdown from rolling maximum.
    
    Drawdown = (Price - RollingMax) / RollingMax
    
    Deeper drawdown → More undervalued (potentially)
    
    Parameters
    ----------
    prices : np.ndarray
        Price series
    lookback : int
        Lookback for rolling maximum (default: 252)
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Dict]
        - drawdown: Drawdown series (negative values)
        - dd_percentile: Percentile rank of drawdown (high = deep drawdown)
        - meta: Metadata
    """
    prices = np.asarray(prices, dtype=np.float64)
    n = len(prices)
    
    # Rolling maximum
    rolling_max = np.full(n, np.nan)
    for i in range(n):
        start = max(0, i - lookback + 1)
        rolling_max[i] = np.nanmax(prices[start:i + 1])
    
    # Drawdown
    drawdown = (prices - rolling_max) / np.maximum(rolling_max, 1e-10)
    
    # Percentile rank (deeper drawdown = higher percentile)
    dd_percentile = np.full(n, np.nan)
    
    for i in range(lookback, n):
        current_dd = drawdown[i]
        hist_dd = drawdown[:i]
        valid_hist = hist_dd[~np.isnan(hist_dd)]
        
        if len(valid_hist) >= lookback // 2:
            # What fraction of historical drawdowns are less severe?
            # More severe (more negative) = higher "undervaluation" score
            pct = np.sum(valid_hist >= current_dd) / len(valid_hist)
            dd_percentile[i] = pct
    
    meta = {
        "lookback": lookback,
        "method": "drawdown",
        "notes": "Drawdown percentile: higher = deeper drawdown = more undervalued"
    }
    
    return drawdown, dd_percentile, meta


def undervaluation_score(
    prices: np.ndarray,
    volumes: Optional[np.ndarray] = None,
    high: Optional[np.ndarray] = None,
    low: Optional[np.ndarray] = None,
    method: str = "combined",
    vwap_window: int = 20,
    zscore_window: int = 50,
    dd_lookback: int = 252,
    percentile_lookback: int = 252
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute normalized undervaluation score U_t ∈ [0, 1].
    
    Higher score = more undervalued = more favorable for DCA.
    
    Parameters
    ----------
    prices : np.ndarray
        Close prices
    volumes : np.ndarray, optional
        Volume data
    high, low : np.ndarray, optional
        High/Low prices for typical price calculation
    method : str
        "vwap_z", "drawdown", or "combined"
    vwap_window : int
        Window for VWAP (default: 20)
    zscore_window : int
        Window for Z-score stats (default: 50)
    dd_lookback : int
        Lookback for drawdown (default: 252)
    percentile_lookback : int
        Lookback for normalization (default: 252)
    
    Returns
    -------
    Tuple[np.ndarray, Dict]
        - U_t: Undervaluation score [0, 1]
        - meta: Metadata
    """
    prices = np.asarray(prices, dtype=np.float64)
    n = len(prices)
    
    # Compute typical price if we have high/low
    if high is not None and low is not None:
        typical_price = (np.asarray(high) + np.asarray(low) + prices) / 3
    else:
        typical_price = prices
    
    scores = {}
    
    # VWAP Z-score component
    if method in ["vwap_z", "combined"]:
        z_t, z_meta = price_vwap_zscore(typical_price, volumes, vwap_window, zscore_window)
        
        # Convert Z-score to [0, 1] score
        # Negative Z = undervalued = high score
        vwap_score = np.full(n, np.nan)
        
        for i in range(percentile_lookback, n):
            if np.isnan(z_t[i]):
                continue
            hist_z = z_t[:i]
            valid = hist_z[~np.isnan(hist_z)]
            if len(valid) >= percentile_lookback // 2:
                # Lower Z = more undervalued = higher score
                pct = np.sum(valid >= z_t[i]) / len(valid)
                vwap_score[i] = pct
        
        scores["vwap_z"] = vwap_score
    
    # Drawdown component
    if method in ["drawdown", "combined"]:
        _, dd_pct, _ = drawdown_score(prices, dd_lookback)
        scores["drawdown"] = dd_pct
    
    # Combine scores
    if method == "combined":
        U_t = np.full(n, np.nan)
        
        for i in range(n):
            valid_scores = []
            
            if "vwap_z" in scores and not np.isnan(scores["vwap_z"][i]):
                valid_scores.append(scores["vwap_z"][i])
            
            if "drawdown" in scores and not np.isnan(scores["drawdown"][i]):
                valid_scores.append(scores["drawdown"][i])
            
            if valid_scores:
                # Equal weight average
                U_t[i] = np.mean(valid_scores)
    else:
        # Single method
        U_t = list(scores.values())[0] if scores else np.full(n, np.nan)
    
    meta = {
        "method": method,
        "vwap_window": vwap_window,
        "zscore_window": zscore_window,
        "dd_lookback": dd_lookback,
        "percentile_lookback": percentile_lookback,
        "components": list(scores.keys()),
        "notes": f"Undervaluation score using {method}: higher = more undervalued = favorable"
    }
    
    return U_t, meta


def compute_undervaluation(prices: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Convenience alias for undervaluation_score."""
    return undervaluation_score(prices, **kwargs)
