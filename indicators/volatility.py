import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def realized_vol(
    series: np.ndarray,
    window: int = 21
) -> Tuple[np.ndarray, Dict[str, Any]]:
    series = np.asarray(series, dtype=np.float64)
    n = len(series)
    
    safe_prices = np.maximum(series, 1e-10)
    log_returns = np.diff(np.log(safe_prices))
    
    vol_t = np.full(n, np.nan)
    
    for i in range(window, n):
        ret_window = log_returns[i - window:i]
        valid_rets = ret_window[~np.isnan(ret_window)]
        
        if len(valid_rets) >= window // 2:
            daily_vol = np.std(valid_rets, ddof=1)
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
    vol_series = np.asarray(vol_series, dtype=np.float64)
    n = len(vol_series)
    
    pct_t = np.full(n, np.nan)
    
    for i in range(lookback, n):
        current_vol = vol_series[i]
        if np.isnan(current_vol):
            continue
        
        hist_vol = vol_series[:i]
        valid_hist = hist_vol[~np.isnan(hist_vol)]
        
        if len(valid_hist) < lookback // 2:
            continue
        
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
    vol_t, vol_meta = realized_vol(series, window=vol_window)
    
    pct_t, pct_meta = volatility_percentile(vol_t, lookback=pct_lookback)
    
    V_t = (100.0 - pct_t) / 100.0                 
    
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


def compute_volatility(series: np.ndarray, window: int = 21) -> Tuple[np.ndarray, Dict[str, Any]]:
    return realized_vol(series, window)


def vol_percentile_score(series: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
    return volatility_regime_score(series, **kwargs)
