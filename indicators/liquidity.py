import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def amihud_illiquidity(
    return_series: np.ndarray,
    volume_series: np.ndarray,
    window: int = 21
) -> Tuple[np.ndarray, Dict[str, Any]]:
    return_series = np.asarray(return_series, dtype=np.float64)
    volume_series = np.asarray(volume_series, dtype=np.float64)
    n = len(return_series)
    
    if len(volume_series) != n:
        raise ValueError("return_series and volume_series must have same length")
    
    safe_volume = np.maximum(volume_series, 1.0)
    daily_illiq = np.abs(return_series) / safe_volume
    
    daily_illiq = np.where(np.isfinite(daily_illiq), daily_illiq, np.nan)
    
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
    price_series = np.asarray(price_series, dtype=np.float64)
    n = len(price_series)
    
    if volume_series is None:
        L_t = np.full(n, 0.5)           
        meta = {
            "window_used": window,
            "n_obs": n,
            "method": "neutral_fallback",
            "seed": None,
            "notes": "No volume data — using neutral 0.5"
        }
        return L_t, meta
    
    volume_series = np.asarray(volume_series, dtype=np.float64)
    
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
    
    safe_prices = np.maximum(price_series, 1e-10)
    returns = np.diff(np.log(safe_prices))
    returns = np.insert(returns, 0, 0)                
    
    ILL_t, ill_meta = amihud_illiquidity(returns, volume_series, window=window)
    
    L_t = np.full(n, np.nan)
    
    for i in range(pct_lookback, n):
        current_ill = ILL_t[i]
        if np.isnan(current_ill):
            continue
        
        hist_ill = ILL_t[:i]
        valid_hist = hist_ill[~np.isnan(hist_ill)]
        
        if len(valid_hist) < pct_lookback // 2:
            continue
        
        pct_worse = np.sum(valid_hist >= current_ill) / len(valid_hist)
        L_t[i] = pct_worse                                                   
    
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
    return liquidity_score(price_series, volume_series, **kwargs)
