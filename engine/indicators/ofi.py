"""
Order Flow Imbalance (OFI) — Phase A microstructure indicator.

Uses a signed-volume proxy:
    OFI_bar = sign(close_t − close_{t-1}) × volume_t

Rolling sum over *window* bars is computed, then normalised through the
expanding-ECDF → inverse-normal → sigmoid pipeline from normalization.py.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
import logging

from engine.indicators.normalization import expanding_ecdf_sigmoid

logger = logging.getLogger(__name__)

def _signed_volume(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Compute signed volume proxy: sign(Δclose) × volume."""
    delta = np.diff(close, prepend=close[0])
    sign = np.sign(delta)
    return sign * volume

def compute_ofi(
    df: pd.DataFrame,
    window: int = 20,
    normalize: bool = True,
    norm_k: float = 1.0,
    norm_polarity: int = 1,
    min_obs: int = 100,
) -> pd.Series:
    """Return OFI time series (per-bar) aligned to df.index.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``close`` and ``volume`` columns (case-insensitive).
    window : int
        Rolling-sum window for OFI accumulation.
    normalize : bool
        If True, apply expanding_ecdf_sigmoid to the raw rolling OFI.
    norm_k : float
        Sigmoid steepness for the normalisation step.
    norm_polarity : int
        +1 → higher OFI is favorable;  -1 → invert.
    min_obs : int
        Warm-up observations for expanding ECDF.

    Returns
    -------
    pd.Series
        Normalised OFI ∈ (0, 1), or raw rolling OFI if normalize=False.
    """
    col_map = {c.lower(): c for c in df.columns}
    close_col = col_map.get("close", "close")
    volume_col = col_map.get("volume", "volume")

    close = df[close_col].values.astype(np.float64)
    volume = df[volume_col].values.astype(np.float64)

    signed_vol = _signed_volume(close, volume)

    ofi_raw = pd.Series(signed_vol, index=df.index, name="ofi_raw")
    ofi_rolling = ofi_raw.rolling(window=window, min_periods=window).sum()

    if not normalize:
        return ofi_rolling

    ofi_norm = expanding_ecdf_sigmoid(
        ofi_rolling,
        k=norm_k,
        polarity=norm_polarity,
        min_obs=min_obs,
    )
    ofi_norm.name = "ofi"
    return ofi_norm

def compute_ofi_reversal(
    df: pd.DataFrame,
    window: int = 20,
    min_obs: int = 100,
) -> pd.Series:
    """OFI reversal signal for Exit Score.

    Returns the *inverted* normalised OFI — high values indicate
    that order flow has turned negative (selling pressure), which
    feeds the exit-score formula.
    """
    return compute_ofi(
        df,
        window=window,
        normalize=True,
        norm_polarity=-1,
        min_obs=min_obs,
    )
