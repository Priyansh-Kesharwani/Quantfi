import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    logger.warning("pywt not available. Using R/S fallback for Hurst estimation.")


def _rescaled_range_hurst(series: np.ndarray, max_k: Optional[int] = None) -> float:
    n = len(series)
    if n < 20:
        return 0.5                                                    
    
    if max_k is None:
        max_k = int(np.log2(n)) - 1
    
    max_k = max(2, min(max_k, int(np.log2(n)) - 1))
    
    rs_values = []
    n_values = []
    
    for k in range(max_k):
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
            
            mean_sub = np.mean(subseries)
            std_sub = np.std(subseries, ddof=1)
            
            if std_sub < 1e-10:
                continue
            
            cumdev = np.cumsum(subseries - mean_sub)
            
            r = np.max(cumdev) - np.min(cumdev)
            rs = r / std_sub
            rs_list.append(rs)
        
        if rs_list:
            rs_values.append(np.mean(rs_list))
            n_values.append(n_sub)
    
    if len(rs_values) < 2:
        return 0.5                     
    
    log_n = np.log(n_values)
    log_rs = np.log(rs_values)
    
    n_points = len(log_n)
    sum_x = np.sum(log_n)
    sum_y = np.sum(log_rs)
    sum_xy = np.sum(log_n * log_rs)
    sum_x2 = np.sum(log_n ** 2)
    
    denom = n_points * sum_x2 - sum_x ** 2
    if abs(denom) < 1e-10:
        return 0.5
    
    H_raw = (n_points * sum_xy - sum_x * sum_y) / denom
    shrink = min(1.0, np.sqrt(n / 512.0))
    H = 0.5 + (H_raw - 0.5) * shrink
    return float(np.clip(H, 0.01, 0.99))


def _wavelet_hurst(series: np.ndarray, wavelet: str = 'db4', 
                   max_level: Optional[int] = None) -> float:
    if not PYWT_AVAILABLE:
        raise ImportError("pywt required for wavelet Hurst estimation")
    
    n = len(series)
    if n < 32:
        return 0.5                     
    
    if max_level is None:
        max_level = int(np.log2(n)) - 2
    
    max_level = max(2, min(max_level, pywt.dwt_max_level(n, wavelet)))
    
    coeffs = pywt.wavedec(series, wavelet, level=max_level)
    
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
        return 0.5                     
    
    scales = np.array(scales)
    log_var = np.log(variances)
    
    n_points = len(scales)
    sum_x = np.sum(scales)
    sum_y = np.sum(log_var)
    sum_xy = np.sum(scales * log_var)
    sum_x2 = np.sum(scales ** 2)
    
    denom = n_points * sum_x2 - sum_x ** 2
    if abs(denom) < 1e-10:
        return 0.5
    
    slope = (n_points * sum_xy - sum_x * sum_y) / denom
    
    H = (slope / np.log(2) + 1) / 2
    
    return float(np.clip(H, 0.01, 0.99))


def estimate_hurst(
    series: np.ndarray,
    window: Optional[int] = None,
    method: str = "auto"
) -> Tuple[np.ndarray, Dict[str, Any]]:
    series = np.asarray(series, dtype=np.float64)
    n = len(series)
    
    if window is None:
        window = min(252, n // 2)
    
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
    
    safe_prices = np.maximum(series, 1e-10)
    log_returns = np.diff(np.log(safe_prices))
    
    H_t = np.full(n, np.nan)
    
    for i in range(window, n):
        ret_window = log_returns[i - window:i]
        
        if len(ret_window) < window:
            continue
        
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
        "seed": None,                 
        "notes": notes if notes else "Estimation completed successfully"
    }
    
    return H_t, meta


hurst_exponent = estimate_hurst
