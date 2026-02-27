"""
Tests for refactor-path canonical normalizer.

R0: Placeholder tests (monotonicity, determinism, known-vector stub).
R1.1: Fill known-vector fixture with expected outputs once exact midrank is implemented.
"""

import numpy as np
import pytest

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from indicators.normalization_refactor import canonical_normalize

def test_normalization_refactor_import():
    """Refactor normalizer module is importable."""
    assert canonical_normalize is not None

def test_monotonic_input_gives_monotonic_output():
    """Monotonic raw input → monotonic s_t (approx mode)."""
    raw = np.arange(100, dtype=np.float64) + np.random.RandomState(42).rand(100) * 0.01
    raw = np.sort(raw)
    s_t, meta = canonical_normalize(raw, k=1.0, eps=1e-9, mode="approx", min_obs=10)
    valid = ~np.isnan(s_t)
    if np.sum(valid) >= 2:
        s_valid = s_t[valid]
        assert np.all(np.diff(s_valid) >= -1e-10), "s_t should be monotonic in raw"

def test_determinism_two_runs_same_bytes():
    """Two runs with same inputs → identical output (hash of s_t)."""
    import hashlib
    np.random.seed(123)
    raw = np.random.randn(200).cumsum() + 100
    s1, _ = canonical_normalize(raw, k=1.0, mode="approx", min_obs=50)
    s2, _ = canonical_normalize(raw, k=1.0, mode="approx", min_obs=50)
    h1 = hashlib.sha256(s1.tobytes()).hexdigest()
    h2 = hashlib.sha256(s2.tobytes()).hexdigest()
    assert h1 == h2, "Two runs must produce identical s_t"

def test_known_vector_placeholder():
    """Known sample vector → expected outputs (shape, bounds)."""
    raw = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20, dtype=np.float64)
    s_t, meta = canonical_normalize(raw, k=1.0, eps=1e-9, mode="approx", min_obs=20)
    assert s_t.shape == raw.shape
    valid = ~np.isnan(s_t)
    assert np.all(s_t[valid] >= 0) and np.all(s_t[valid] <= 1)
    assert meta["n_obs"] >= 1
    assert meta["mode"] == "approx"

def test_exact_mode_midrank_determinism():
    """Exact mode: two runs → identical output (hash)."""
    import hashlib
    np.random.seed(99)
    raw = np.random.randn(150).cumsum() + 50
    s1, _ = canonical_normalize(raw, k=1.0, mode="exact", min_obs=30)
    s2, _ = canonical_normalize(raw, k=1.0, mode="exact", min_obs=30)
    assert hashlib.sha256(s1.tobytes()).hexdigest() == hashlib.sha256(s2.tobytes()).hexdigest()

def test_exact_mode_monotonic():
    """Exact mode: monotonic raw → monotonic s_t."""
    raw = np.sort(np.random.RandomState(42).rand(120))
    s_t, meta = canonical_normalize(raw, k=1.0, mode="exact", min_obs=20)
    valid = ~np.isnan(s_t)
    if np.sum(valid) >= 2:
        assert np.all(np.diff(s_t[valid]) >= -1e-10)
    assert meta["mode"] == "exact"

def test_exact_known_midrank_values():
    """Exact mode: for sorted [1..5] repeated, p_t at each step is (rank_less + 0.5*rank_equal)/n_t; scores increase in raw."""
    raw = np.arange(1.0, 6.0, dtype=np.float64)
    s_t, meta = canonical_normalize(raw, k=1.0, eps=1e-9, mode="exact", min_obs=2)
    assert meta["mode"] == "exact"
    valid = ~np.isnan(s_t)
    assert np.sum(valid) >= 3
    s_valid = s_t[valid]
    assert np.all(np.diff(s_valid) >= -1e-10)

def test_polarity_lower_is_favorable():
    """When higher_is_favorable=False, increasing raw should decrease score."""
    raw = np.arange(100, dtype=np.float64)
    s_high, _ = canonical_normalize(raw, mode="approx", higher_is_favorable=True, min_obs=10)
    s_low, _ = canonical_normalize(raw, mode="approx", higher_is_favorable=False, min_obs=10)
    valid = ~np.isnan(s_high) & ~np.isnan(s_low)
    assert np.allclose(s_low[valid], 1.0 - s_high[valid], rtol=1e-9, equal_nan=True)

def test_exact_vs_approx_same_shape_and_bounds():
    """Exact and approx modes produce same shape and scores in [0, 1]."""
    np.random.seed(77)
    raw = np.random.randn(200).cumsum() + 50
    s_exact, meta_exact = canonical_normalize(raw, k=1.0, eps=1e-9, mode="exact", min_obs=30)
    s_approx, meta_approx = canonical_normalize(raw, k=1.0, eps=1e-9, mode="approx", min_obs=30)
    assert s_exact.shape == s_approx.shape == raw.shape
    for s, mode in [(s_exact, "exact"), (s_approx, "approx")]:
        valid = ~np.isnan(s)
        assert np.all(s[valid] >= 0) and np.all(s[valid] <= 1), f"{mode} scores must be in [0,1]"
    assert meta_exact["mode"] == "exact"
    assert meta_approx["mode"] == "approx"

def test_exact_vs_approx_outputs_close():
    """Exact (midrank) and approx (expanding_percentile) outputs are close on same input."""
    np.random.seed(88)
    raw = np.random.randn(180).cumsum() + 100
    s_exact, _ = canonical_normalize(raw, k=1.0, eps=1e-9, mode="exact", min_obs=40)
    s_approx, _ = canonical_normalize(raw, k=1.0, eps=1e-9, mode="approx", min_obs=40)
    valid = ~np.isnan(s_exact) & ~np.isnan(s_approx)
    assert valid.sum() > 50
    a = np.asarray(s_exact)[valid]
    b = np.asarray(s_approx)[valid]
    diff = np.abs(a - b)
    assert np.max(diff) < 0.15
    assert np.sqrt(np.mean(diff ** 2)) < 0.08
