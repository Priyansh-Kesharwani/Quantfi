"""
Bot feature API: OFI, Hawkes, LDC, ATR.

Delegates to indicators where applicable; implements ATR in-module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Union

from indicators.ofi import compute_ofi as _compute_ofi
from indicators.hawkes import estimate_hawkes as _estimate_hawkes
from indicators.ldc import LDC as _LDC

def compute_ofi(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Order Flow Imbalance (rolling signed-volume proxy), normalized to (0,1).

    Delegates to indicators.ofi.compute_ofi with default normalize=True.
    """
    return _compute_ofi(df, window=window, normalize=True, min_obs=min(100, max(window, len(df) // 4)))

def estimate_hawkes(
    event_times_dict: Dict[str, np.ndarray],
    timestamps: Union[np.ndarray, pd.Index],
    decay: float = 1.0,
) -> pd.Series:
    """
    Hawkes intensity λ(t) on a regular timestamp grid.

    Uses tick backend if available, else incremental/scipy MLE. Returns only
    the intensity Series (not meta).
    """
    ts = np.asarray(timestamps, dtype=np.float64).ravel()
    intensity, _ = _estimate_hawkes(event_times_dict, ts, decay=decay)
    return intensity

LDC = _LDC

def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Average True Range: ATR = rolling mean of TR over `window`.

    TR_t = max(high - low, |high - prev_close|, |low - prev_close|).
    First row uses high-low only (no prev_close). Output aligned to df.index.
    """
    col = {c.lower(): c for c in df.columns}
    high_col = col.get("high", "high")
    low_col = col.get("low", "low")
    close_col = col.get("close", "close")
    if high_col not in df.columns or low_col not in df.columns or close_col not in df.columns:
        raise ValueError("df must contain high, low, close columns")
    high = df[high_col].astype(float)
    low = df[low_col].astype(float)
    close = df[close_col].astype(float)
    prev_close = close.shift(1)
    tr = np.maximum(
        high - low,
        np.maximum(
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ),
    )
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    atr = tr.rolling(window=window, min_periods=1).mean()
    atr.name = "atr"
    return atr
