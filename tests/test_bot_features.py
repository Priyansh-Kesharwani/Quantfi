"""
Unit tests for bot.features: OFI, Hawkes, LDC, ATR.

Uses synthetic FBM/OU/Hawkes fixtures; determinism with fixed seed.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bot.features import compute_ofi, estimate_hawkes, compute_atr, LDC

try:
    from tests.fixtures import ou_series, hawkes_events
except ImportError:
    ou_series = None
    hawkes_events = None

def _ou_bar_df(n: int = 120, seed: int = 42) -> pd.DataFrame:
    """Small bar DataFrame with open, high, low, close, volume."""
    if ou_series is not None:
        return ou_series(n=n, seed=seed)
    np.random.seed(seed)
    x = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = x + np.abs(np.random.randn(n)) * 0.3
    low = x - np.abs(np.random.randn(n)) * 0.3
    return pd.DataFrame(
        {"open": np.roll(x, 1), "high": high, "low": low, "close": x, "volume": np.abs(np.random.randn(n) * 1e6 + 5e6)},
        index=pd.date_range("2020-01-01", periods=n, freq="min", tz="UTC"),
    )

def test_compute_ofi_returns_series():
    """compute_ofi returns a pd.Series aligned to df.index."""
    df = _ou_bar_df(150, seed=1)
    ofi = compute_ofi(df, window=20)
    assert isinstance(ofi, pd.Series)
    assert len(ofi) == len(df)
    assert ofi.index.equals(df.index)

def test_compute_ofi_deterministic():
    """Same df and window produce identical OFI (fixed seed in fixture)."""
    df = _ou_bar_df(150, seed=99)
    o1 = compute_ofi(df, window=20)
    o2 = compute_ofi(df, window=20)
    valid = ~o1.isna() & ~o2.isna()
    assert np.allclose(o1[valid].values, o2[valid].values, equal_nan=True)

def test_estimate_hawkes_returns_series():
    """estimate_hawkes returns a pd.Series (intensity)."""
    events = np.array([1.0, 2.0, 3.5, 5.0, 7.0, 10.0])
    timestamps = np.linspace(0, 12, 25)
    intensity = estimate_hawkes({"trades": events}, timestamps, decay=1.0)
    assert isinstance(intensity, pd.Series)
    assert len(intensity) == len(timestamps)

def test_estimate_hawkes_deterministic():
    """Same events and timestamps produce identical intensity."""
    if hawkes_events is None:
        events = np.array([1.0, 2.5, 4.0, 6.0, 8.0])
    else:
        events = hawkes_events(T=20, seed=42)
    timestamps = np.linspace(0, 25, 50)
    i1 = estimate_hawkes({"trades": events}, timestamps, decay=1.0)
    i2 = estimate_hawkes({"trades": events}, timestamps, decay=1.0)
    assert np.allclose(i1.values, i2.values, equal_nan=True)

def test_ldc_fit_score_skeleton():
    """LDC.fit(templates) and LDC.score(x) work; score in (0,1)."""
    ldc = LDC(kappa=1.0)
    templates = {"bull": np.array([1.0, 2.0]), "bear": np.array([-1.0, -2.0])}
    ldc.fit(templates)
    s = ldc.score(np.array([1.0, 2.0]))
    assert 0 <= s <= 1
    assert s > 0.5
    s_bear = ldc.score(np.array([-1.0, -2.0]))
    assert s_bear < 0.5

def test_ldc_score_requires_fit():
    """LDC.score() without fit raises."""
    ldc = LDC()
    with pytest.raises(RuntimeError, match="fitted"):
        ldc.score(np.array([1.0, 0.0]))

def test_compute_atr_returns_series():
    """compute_atr returns Series aligned to df.index."""
    df = _ou_bar_df(50, seed=3)
    atr = compute_atr(df, window=14)
    assert isinstance(atr, pd.Series)
    assert len(atr) == len(df)
    assert atr.index.equals(df.index)
    assert atr.name == "atr"

def test_compute_atr_positive():
    """ATR values are non-negative."""
    df = _ou_bar_df(50, seed=4)
    atr = compute_atr(df, window=14)
    assert (atr.dropna() >= 0).all()

def test_compute_atr_requires_ohlc():
    """compute_atr raises if high/low/close missing."""
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError, match="high|low|close"):
        compute_atr(df, window=14)

def test_compute_atr_deterministic():
    """Same df and window produce identical ATR."""
    df = _ou_bar_df(80, seed=7)
    a1 = compute_atr(df, window=14)
    a2 = compute_atr(df, window=14)
    assert np.allclose(a1.values, a2.values, equal_nan=True)
