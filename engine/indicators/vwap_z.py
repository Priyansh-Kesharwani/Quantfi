import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def _compute_rolling_vwap(
    price: np.ndarray,
    volume: np.ndarray,
    window: int
) -> np.ndarray:
    n = len(price)
    vwap = np.full(n, np.nan)
    
    for i in range(window, n + 1):
        p = price[i - window:i]
        v = volume[i - window:i]
        
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
    n = len(price)
    z_score = np.full(n, np.nan)
    
    for i in range(vol_window, n):
        if np.isnan(benchmark[i]) or np.isnan(price[i]):
            continue
        
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
    price_series = np.asarray(price_series, dtype=np.float64)
    n = len(price_series)
    
    notes = ""
    
    if volume_series is not None:
        volume_series = np.asarray(volume_series, dtype=np.float64)
        
        valid_volume = volume_series[~np.isnan(volume_series)]
        if len(valid_volume) > n // 2 and np.sum(valid_volume) > 0:
            benchmark = _compute_rolling_vwap(price_series, volume_series, window)
            method = "vwap"
        else:
            benchmark = _compute_rolling_sma(price_series, window)
            method = "sma_fallback"
            notes = "Volume data insufficient, using SMA as VWAP proxy"
    else:
        benchmark = _compute_rolling_sma(price_series, window)
        method = "sma_fallback"
        notes = "No volume data provided — using SMA proxy"
    
    vol_window = window                                  
    Z_t = _compute_z_score(price_series, benchmark, vol_window)
    
    meta = {
        "window_used": window,
        "n_obs": n,
        "method": method,
        "seed": None,                 
        "notes": notes if notes else f"VWAP Z-score computed with {window}-day window"
    }
    
    return Z_t, meta


def vwap_undervaluation_score(
    price_series: np.ndarray,
    volume_series: Optional[np.ndarray] = None,
    window: int = 20
) -> Tuple[np.ndarray, Dict[str, Any]]:
    Z_t, meta = compute_vwap_z(price_series, volume_series, window)
    
    U_t = -Z_t
    
    meta["notes"] = meta.get("notes", "") + " | Polarity inverted for undervaluation score"
    
    return U_t, meta
