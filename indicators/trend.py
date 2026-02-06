"""
Trend Strength Indicator (T_t)

Measures the strength and clarity of the current price trend.
Used in the Opportunity block of the composite score.

Methods Supported:
- EMA slope (rate of change of exponential moving average)
- ADX (Average Directional Index)
- MACD histogram analysis
- Combined (weighted average of multiple methods)

For the Opportunity calculation:
- Strong uptrend → Higher T_t → More favorable for DCA (buying into strength)
- Weak/no trend → Lower T_t → Less confident about entry timing

Author: Phase 2 Implementation
Date: 2026-02-07
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def ema(series: np.ndarray, span: int) -> np.ndarray:
    """
    Compute Exponential Moving Average.
    
    EMA_t = α × price_t + (1-α) × EMA_{t-1}
    where α = 2/(span+1)
    """
    alpha = 2.0 / (span + 1)
    ema_values = np.full_like(series, np.nan, dtype=np.float64)
    
    # Find first valid value
    first_valid = np.argmax(~np.isnan(series))
    if np.isnan(series[first_valid]):
        return ema_values
    
    ema_values[first_valid] = series[first_valid]
    
    for i in range(first_valid + 1, len(series)):
        if not np.isnan(series[i]):
            ema_values[i] = alpha * series[i] + (1 - alpha) * ema_values[i - 1]
        else:
            ema_values[i] = ema_values[i - 1]
    
    return ema_values


def ema_slope(
    prices: np.ndarray,
    short_span: int = 12,
    long_span: int = 26,
    slope_window: int = 5
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute trend strength from EMA slope.
    
    The slope of the short-term EMA relative to its own values indicates
    trend strength and direction.
    
    Parameters
    ----------
    prices : np.ndarray
        Price series
    short_span : int
        Short EMA span (default: 12)
    long_span : int
        Long EMA span (default: 26)
    slope_window : int
        Window for computing slope (default: 5)
    
    Returns
    -------
    Tuple[np.ndarray, Dict]
        - slope_t: Normalized EMA slope
        - meta: Metadata
    """
    prices = np.asarray(prices, dtype=np.float64)
    n = len(prices)
    
    # Compute EMAs
    ema_short = ema(prices, short_span)
    ema_long = ema(prices, long_span)
    
    # EMA difference (like MACD line)
    ema_diff = ema_short - ema_long
    
    # Compute slope of the short EMA
    slope_t = np.full(n, np.nan)
    
    for i in range(slope_window, n):
        window_ema = ema_short[i - slope_window:i + 1]
        if np.any(np.isnan(window_ema)):
            continue
        
        # Simple linear regression slope
        x = np.arange(slope_window + 1)
        slope = np.polyfit(x, window_ema, 1)[0]
        
        # Normalize by current price level
        if prices[i] > 0:
            slope_t[i] = slope / prices[i]
    
    meta = {
        "short_span": short_span,
        "long_span": long_span,
        "slope_window": slope_window,
        "method": "ema_slope",
        "notes": "Normalized EMA slope (positive = uptrend)"
    }
    
    return slope_t, meta


def adx_indicator(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int = 14
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Compute Average Directional Index (ADX).
    
    ADX measures trend strength regardless of direction.
    Values > 25 typically indicate a strong trend.
    
    Parameters
    ----------
    high : np.ndarray
        High prices
    low : np.ndarray
        Low prices
    close : np.ndarray
        Close prices
    window : int
        ADX calculation window (default: 14)
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]
        - ADX: Average Directional Index
        - +DI: Positive directional indicator
        - -DI: Negative directional indicator
        - meta: Metadata
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    
    # True Range
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )
    tr = np.insert(tr, 0, high[0] - low[0])
    
    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    
    # +DM and -DM
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    
    # Smoothed values using Wilder's smoothing
    atr = np.full(n, np.nan)
    plus_dm_smooth = np.full(n, np.nan)
    minus_dm_smooth = np.full(n, np.nan)
    
    # Initialize
    atr[window] = np.mean(tr[:window + 1])
    plus_dm_smooth[window] = np.mean(plus_dm[:window + 1])
    minus_dm_smooth[window] = np.mean(minus_dm[:window + 1])
    
    # Wilder's smoothing
    for i in range(window + 1, n):
        atr[i] = (atr[i-1] * (window - 1) + tr[i]) / window
        plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (window - 1) + plus_dm[i]) / window
        minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (window - 1) + minus_dm[i]) / window
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / np.maximum(atr, 1e-10)
    minus_di = 100 * minus_dm_smooth / np.maximum(atr, 1e-10)
    
    # DX
    dx = 100 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-10)
    
    # ADX (smoothed DX)
    adx = np.full(n, np.nan)
    adx[2 * window] = np.nanmean(dx[window:2 * window + 1])
    
    for i in range(2 * window + 1, n):
        if not np.isnan(adx[i-1]) and not np.isnan(dx[i]):
            adx[i] = (adx[i-1] * (window - 1) + dx[i]) / window
    
    meta = {
        "window": window,
        "method": "adx",
        "notes": "ADX: 0-25 weak, 25-50 strong, 50-75 very strong, 75-100 extremely strong"
    }
    
    return adx, plus_di, minus_di, meta


def macd_histogram(
    prices: np.ndarray,
    fast_span: int = 12,
    slow_span: int = 26,
    signal_span: int = 9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Compute MACD and histogram.
    
    MACD Line = EMA(fast) - EMA(slow)
    Signal Line = EMA(MACD, signal_span)
    Histogram = MACD - Signal
    
    Parameters
    ----------
    prices : np.ndarray
        Price series
    fast_span : int
        Fast EMA span (default: 12)
    slow_span : int
        Slow EMA span (default: 26)
    signal_span : int
        Signal line EMA span (default: 9)
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]
        - macd: MACD line
        - signal: Signal line
        - histogram: MACD histogram
        - meta: Metadata
    """
    prices = np.asarray(prices, dtype=np.float64)
    
    ema_fast = ema(prices, fast_span)
    ema_slow = ema(prices, slow_span)
    
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal_span)
    histogram = macd_line - signal_line
    
    meta = {
        "fast_span": fast_span,
        "slow_span": slow_span,
        "signal_span": signal_span,
        "method": "macd",
        "notes": "MACD histogram: positive = bullish momentum, negative = bearish"
    }
    
    return macd_line, signal_line, histogram, meta


def trend_strength_score(
    prices: np.ndarray,
    high: Optional[np.ndarray] = None,
    low: Optional[np.ndarray] = None,
    method: str = "combined",
    ema_short: int = 12,
    ema_long: int = 26,
    adx_window: int = 14,
    lookback: int = 252
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute normalized trend strength score T_t ∈ [0, 1].
    
    Higher score indicates stronger, more favorable trend for DCA.
    
    Parameters
    ----------
    prices : np.ndarray
        Close prices
    high : np.ndarray, optional
        High prices (required for ADX)
    low : np.ndarray, optional
        Low prices (required for ADX)
    method : str
        "ema_slope", "adx", "macd", or "combined"
    ema_short, ema_long : int
        EMA parameters
    adx_window : int
        ADX calculation window
    lookback : int
        Lookback for normalization
    
    Returns
    -------
    Tuple[np.ndarray, Dict]
        - T_t: Trend strength score [0, 1]
        - meta: Metadata
    """
    prices = np.asarray(prices, dtype=np.float64)
    n = len(prices)
    
    scores = {}
    
    # EMA Slope component
    if method in ["ema_slope", "combined"]:
        slope, _ = ema_slope(prices, ema_short, ema_long)
        # Normalize to [0, 1] using expanding percentile
        slope_score = np.full(n, np.nan)
        for i in range(lookback, n):
            if np.isnan(slope[i]):
                continue
            hist = slope[:i]
            valid = hist[~np.isnan(hist)]
            if len(valid) >= lookback // 2:
                # Percentile (higher slope = higher score)
                pct = np.sum(valid <= slope[i]) / len(valid)
                slope_score[i] = pct
        scores["ema_slope"] = slope_score
    
    # ADX component
    if method in ["adx", "combined"] and high is not None and low is not None:
        adx, plus_di, minus_di, _ = adx_indicator(high, low, prices, adx_window)
        # ADX > 25 is strong trend
        # Normalize: ADX/100, clip to [0, 1]
        adx_score = np.clip(adx / 100, 0, 1)
        # Consider direction: +DI > -DI is uptrend
        direction = np.where(plus_di > minus_di, 1.0, 0.5)
        # Combined: trend strength weighted by favorable direction
        adx_final = adx_score * direction
        scores["adx"] = adx_final
    
    # MACD component
    if method in ["macd", "combined"]:
        _, _, histogram, _ = macd_histogram(prices, ema_short, ema_long)
        # Normalize histogram to [0, 1]
        macd_score = np.full(n, np.nan)
        for i in range(lookback, n):
            if np.isnan(histogram[i]):
                continue
            hist_vals = histogram[:i]
            valid = hist_vals[~np.isnan(hist_vals)]
            if len(valid) >= lookback // 2:
                pct = np.sum(valid <= histogram[i]) / len(valid)
                macd_score[i] = pct
        scores["macd"] = macd_score
    
    # Combine scores
    if method == "combined":
        # Weight components
        T_t = np.full(n, np.nan)
        for i in range(n):
            valid_scores = []
            weights = []
            
            if "ema_slope" in scores and not np.isnan(scores["ema_slope"][i]):
                valid_scores.append(scores["ema_slope"][i])
                weights.append(0.3)
            
            if "adx" in scores and not np.isnan(scores["adx"][i]):
                valid_scores.append(scores["adx"][i])
                weights.append(0.4)
            
            if "macd" in scores and not np.isnan(scores["macd"][i]):
                valid_scores.append(scores["macd"][i])
                weights.append(0.3)
            
            if valid_scores:
                weights = np.array(weights) / sum(weights)
                T_t[i] = np.average(valid_scores, weights=weights)
    else:
        # Single method
        T_t = list(scores.values())[0] if scores else np.full(n, np.nan)
    
    meta = {
        "method": method,
        "ema_short": ema_short,
        "ema_long": ema_long,
        "adx_window": adx_window,
        "lookback": lookback,
        "components": list(scores.keys()),
        "notes": f"Trend strength score using {method} method"
    }
    
    return T_t, meta


def compute_trend(prices: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Convenience alias for trend_strength_score."""
    return trend_strength_score(prices, **kwargs)
