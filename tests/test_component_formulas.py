"""
Tests for refactor-path indicator components.

Each indicator returns (pd.Series, dict meta) with
meta: {name, window, n_obs, unit, polarity, warnings}.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


META_KEYS = {"name", "window", "n_obs", "unit", "polarity", "warnings"}


def test_component_contract_placeholder():
    """Expected meta contract for all refactor components."""
    assert "name" in META_KEYS and "window" in META_KEYS


def test_normalizer_used_by_refactor_path():
    """Refactor path uses normalization_refactor (smoke)."""
    from indicators.normalization_refactor import canonical_normalize
    raw = np.random.RandomState(42).randn(100).cumsum()
    s, meta = canonical_normalize(raw, mode="approx", min_obs=20)
    assert isinstance(s, np.ndarray)
    assert isinstance(meta, dict)
    assert "method" in meta and meta["method"] == "canonical_normalize"


def test_ofi_refactor_returns_series_meta():
    """OFI refactor returns (Series, meta) with required keys."""
    from tests.fixtures import fbm_series
    from indicators.refactor_components import ofi_refactor
    df = fbm_series(n=150, H=0.6, seed=42)
    series, meta = ofi_refactor(df, window=20, min_obs=30)
    assert isinstance(series, pd.Series)
    assert series.name == "OFI"
    assert META_KEYS.issubset(meta.keys())
    assert meta["name"] == "OFI"
    assert meta["window"] == 20
    valid = series.dropna()
    assert len(valid) <= len(df)
    if len(valid) >= 2:
        assert (valid >= 0).all() and (valid <= 1).all()


def test_vwap_z_refactor_returns_series_meta():
    """VWAP Z refactor returns (Series, meta)."""
    from indicators.refactor_components import vwap_z_refactor
    np.random.seed(42)
    price = 100 + np.cumsum(np.random.randn(100) * 0.5)
    vol = np.abs(np.random.randn(100)) * 1e6 + 1e6
    series, meta = vwap_z_refactor(price, vol, vol_window=20)
    assert isinstance(series, pd.Series)
    assert META_KEYS.issubset(meta.keys())
    assert meta["name"] == "VWAP_Z"


def test_hurst_refactor_returns_series_meta():
    """Hurst refactor returns (Series, meta)."""
    from indicators.refactor_components import hurst_refactor
    from tests.fixtures import fbm_series
    df = fbm_series(n=300, H=0.7, seed=43)
    series, meta = hurst_refactor(df["close"].values, window=100, method="rs")
    assert isinstance(series, pd.Series)
    assert META_KEYS.issubset(meta.keys())
    assert meta["name"] == "Hurst"
    valid = series.dropna()
    if len(valid) > 0:
        assert (valid >= 0).all() and (valid <= 1).all()


def test_hawkes_refactor_returns_series_meta():
    """Hawkes refactor returns (Series, meta)."""
    from indicators.refactor_components import hawkes_refactor
    from tests.fixtures import hawkes_events
    events = hawkes_events(seed=44)
    timestamps = np.linspace(0, 50, 200)
    series, meta = hawkes_refactor(
        {"trades": events}, timestamps, dt=0.25, decay=1.0
    )
    assert isinstance(series, pd.Series)
    assert META_KEYS.issubset(meta.keys())
    assert meta["name"] == "Hawkes_lambda"
    assert (series >= 0).all()


def test_ldc_refactor_returns_series_meta():
    """LDC refactor returns (Series, meta)."""
    from indicators.refactor_components import ldc_refactor
    np.random.seed(45)
    bull = np.random.randn(30, 5) + 1
    bear = np.random.randn(30, 5) - 1
    X = np.vstack([bull, bear])
    y = np.array([1] * 30 + [0] * 30)
    series, meta = ldc_refactor(X, labels=y, kappa=1.0)
    assert isinstance(series, pd.Series)
    assert META_KEYS.issubset(meta.keys())
    assert meta["name"] == "LDC"
    assert (series >= 0).all() and (series <= 1).all()


def test_refactor_component_determinism():
    """Refactor OFI: same input + seed → same output hash."""
    import hashlib
    from tests.fixtures import fbm_series
    from indicators.refactor_components import ofi_refactor
    df = fbm_series(n=120, H=0.5, seed=99)
    s1, _ = ofi_refactor(df, window=15, min_obs=20)
    s2, _ = ofi_refactor(df, window=15, min_obs=20)
    h1 = hashlib.sha256(s1.fillna(-1).values.tobytes()).hexdigest()
    h2 = hashlib.sha256(s2.fillna(-1).values.tobytes()).hexdigest()
    assert h1 == h2
