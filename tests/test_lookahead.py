"""
Look-ahead and data correctness: ensure indicators/scores at time t use only data ≤ t.

Tests the normalization layer (expanding_percentile, normalize_to_score): appending
a future value must not change results on the original index. Production scoring
(backend BacktestEngine._compute_rolling_scores) uses only iloc[i] and earlier per bar;
see docs/ARCHITECTURE.md for the documented no-look-ahead design.
"""

import importlib.util
import numpy as np
import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_norm_path = PROJECT_ROOT / "indicators" / "normalization.py"
_spec = importlib.util.spec_from_file_location("normalization", _norm_path)
_norm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_norm)
expanding_percentile = _norm.expanding_percentile
normalize_to_score = _norm.normalize_to_score

def test_expanding_percentile_no_lookahead():
    """expanding_percentile uses historical = series[:t]; appending future must not change pct at t."""
    np.random.seed(42)
    n = 300
    min_obs = 100
    series = np.cumsum(np.random.randn(n)) + 100.0

    pct_full, _ = expanding_percentile(series, min_obs=min_obs)

    extra = np.array([999999.0])
    series_extra = np.concatenate([series, extra])

    pct_extra, _ = expanding_percentile(series_extra, min_obs=min_obs)

    for i in range(min_obs, n):
        a = pct_full[i]
        b = pct_extra[i]
        if np.isnan(a) and np.isnan(b):
            continue
        assert not (np.isnan(a) != np.isnan(b)), f"index {i}: nan mismatch"
        np.testing.assert_allclose(
            a, b, rtol=1e-12, atol=1e-12,
            err_msg=f"At index {i} expanding_percentile must not depend on future data",
        )

def test_normalize_to_score_no_lookahead():
    """normalize_to_score uses expanding windows only; appending future must not change score at t."""
    np.random.seed(43)
    n = 300
    min_obs = 100
    raw = np.cumsum(np.random.randn(n)) + 50.0

    score_full, _ = normalize_to_score(raw, min_obs=min_obs, k=1.0)

    raw_extra = np.concatenate([raw, np.array([1e6])])
    score_extra, _ = normalize_to_score(raw_extra, min_obs=min_obs, k=1.0)

    for i in range(min_obs, n):
        a = score_full[i]
        b = score_extra[i]
        if np.isnan(a) and np.isnan(b):
            continue
        assert not (np.isnan(a) != np.isnan(b)), f"index {i}: nan mismatch"
        np.testing.assert_allclose(
            a, b, rtol=1e-12, atol=1e-12,
            err_msg=f"At index {i} normalize_to_score must not depend on future data",
        )
