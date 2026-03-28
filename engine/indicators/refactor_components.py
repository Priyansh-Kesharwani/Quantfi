"""
Indicator components returning (pd.Series, dict meta) contract.

Each function returns (values: pd.Series, meta: dict) with
meta: {name, window, n_obs, unit, polarity, warnings}.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from engine.indicators.hawkes import estimate_hawkes
from engine.indicators.hurst import estimate_hurst
from engine.indicators.ldc import LDC, build_templates_from_labels
from engine.indicators.ofi import compute_ofi
from engine.indicators.vwap_z import compute_vwap_z


def _standard_meta(
    name: str,
    window: int,
    n_obs: int,
    unit: str = "dimensionless",
    polarity: str = "higher_favorable",
    warnings: Optional[list] = None,
    **extra: Any,
) -> Dict[str, Any]:
    meta = {
        "name": name,
        "window": window,
        "n_obs": n_obs,
        "unit": unit,
        "polarity": polarity,
        "warnings": list(warnings) if warnings else [],
    }
    meta.update(extra)
    return meta


def ofi_refactor(
    df: pd.DataFrame,
    window: int = 20,
    normalize: bool = True,
    min_obs: int = 50,
) -> Tuple[pd.Series, Dict[str, Any]]:
    series = compute_ofi(df, window=window, normalize=normalize, min_obs=min_obs)
    n_obs = int(series.notna().sum())
    meta = _standard_meta(
        name="OFI", window=window, n_obs=n_obs,
        unit="dimensionless", polarity="higher_favorable",
    )
    series = series.rename("OFI")
    return series, meta


def vwap_z_refactor(
    price_series: np.ndarray,
    volume_series: Optional[np.ndarray] = None,
    vol_window: int = 60,
    index: Optional[pd.Index] = None,
) -> Tuple[pd.Series, Dict[str, Any]]:
    z_t, inner_meta = compute_vwap_z(price_series, volume_series, window=vol_window)
    n = len(z_t)
    if index is None:
        index = pd.RangeIndex(n)
    series = pd.Series(z_t, index=index, name="VWAP_Z")
    n_obs = int(np.sum(~np.isnan(z_t)))
    meta = _standard_meta(
        name="VWAP_Z", window=vol_window, n_obs=n_obs,
        unit="z_score", polarity="lower_favorable",
        warnings=[inner_meta.get("notes", "")] if inner_meta.get("notes") else [],
    )
    meta["method"] = inner_meta.get("method", "vwap")
    return series, meta


def hurst_refactor(
    series: np.ndarray,
    window: int = 200,
    method: str = "dfa",
    index: Optional[pd.Index] = None,
) -> Tuple[pd.Series, Dict[str, Any]]:
    actual_method = "wavelet" if method == "dfa" else method
    if actual_method == "dfa":
        actual_method = "rs"
    H_t, inner_meta = estimate_hurst(series, window=window, method=actual_method)
    n = len(H_t)
    if index is None:
        index = pd.RangeIndex(n)
    s = pd.Series(H_t, index=index, name="Hurst")
    n_obs = int(np.sum(~np.isnan(H_t)))
    meta = _standard_meta(
        name="Hurst", window=inner_meta.get("window_used", window), n_obs=n_obs,
        unit="dimensionless", polarity="higher_favorable",
        warnings=[inner_meta.get("notes", "")] if inner_meta.get("notes") else [],
    )
    meta["method"] = inner_meta.get("method", actual_method)
    return s, meta


def hawkes_refactor(
    events: Dict[str, np.ndarray],
    timestamps: np.ndarray,
    dt: float = 1.0,
    decay: float = 1.0,
    index: Optional[pd.Index] = None,
) -> Tuple[pd.Series, Dict[str, Any]]:
    intensity, inner_meta = estimate_hawkes(events, timestamps, dt=dt, decay=decay)
    if index is not None:
        intensity = pd.Series(intensity.values, index=index, name="hawkes_lambda")
    else:
        intensity = intensity.rename("hawkes_lambda")
    n_obs = len(intensity)
    meta = _standard_meta(
        name="Hawkes_lambda", window=0, n_obs=n_obs,
        unit="intensity", polarity="lower_favorable",
    )
    meta["mu"] = inner_meta.get("mu")
    meta["alpha"] = inner_meta.get("alpha")
    meta["beta"] = inner_meta.get("beta")
    meta["n_events"] = inner_meta.get("n_events")
    return intensity, meta


def ldc_refactor(
    features: np.ndarray,
    labels: Optional[np.ndarray] = None,
    templates: Optional[Dict[str, np.ndarray]] = None,
    kappa: float = 1.0,
    feature_window: int = 5,
    index: Optional[pd.Index] = None,
) -> Tuple[pd.Series, Dict[str, Any]]:
    warnings: list = []
    if templates is None:
        if labels is None or features is None:
            raise ValueError("ldc_refactor needs either templates or (features, labels)")
        templates = build_templates_from_labels(features, labels)
    ldc = LDC(kappa=kappa)
    ldc.fit(templates)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    scores = ldc.score_batch(features)
    n = len(scores)
    if index is None:
        index = pd.RangeIndex(n)
    series = pd.Series(scores, index=index, name="LDC")
    meta = _standard_meta(
        name="LDC", window=feature_window, n_obs=n,
        unit="dimensionless", polarity="higher_favorable",
        warnings=warnings,
    )
    meta["kappa"] = kappa
    return series, meta
