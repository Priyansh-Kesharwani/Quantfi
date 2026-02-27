"""
Tests for refactor-path composite (Opp, Gate, RawFavor, CompositeScore, Exit).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _synthetic_components(n: int = 100, seed: int = 42):
    np.random.seed(seed)
    idx = pd.RangeIndex(n)
    return {
        "T_t": pd.Series(np.clip(np.random.rand(n) * 0.5 + 0.25, 0, 1), index=idx),
        "U_t": pd.Series(np.clip(np.random.rand(n) * 0.5 + 0.25, 0, 1), index=idx),
        "H_t": pd.Series(np.clip(np.random.rand(n) * 0.5 + 0.25, 0, 1), index=idx),
        "LDC_t": pd.Series(np.clip(np.random.rand(n) * 0.5 + 0.25, 0, 1), index=idx),
        "O_t": pd.Series(np.clip(np.random.rand(n) * 0.5 + 0.25, 0, 1), index=idx),
        "C_t": pd.Series(np.ones(n) * 0.9, index=idx),
        "L_t": pd.Series(np.ones(n) * 0.9, index=idx),
        "R_t": pd.Series(np.ones(n) * 0.5, index=idx),
        "TBL_flag": pd.Series(np.random.rand(n), index=idx),
        "OFI_rev": pd.Series(np.random.rand(n), index=idx),
        "lambda_decay": pd.Series(np.random.rand(n), index=idx),
    }


def test_composite_refactor_import():
    from indicators.composite_refactor import (
        load_refactor_config,
        compute_composite_score_refactor,
        g_pers_refactor,
    )
    assert load_refactor_config is not None
    assert compute_composite_score_refactor is not None


def test_composite_score_bounded():
    """CompositeScore and Exit are in [0, 100] (Exit may have NaN during normalizer warm-up)."""
    from indicators.composite_refactor import compute_composite_score_refactor
    comp = _synthetic_components(80, seed=1)
    entry, exit_s, breakdown = compute_composite_score_refactor(comp)
    assert (entry >= 0).all() and (entry <= 100).all()
    exit_valid = exit_s.dropna()
    if len(exit_valid) > 0:
        assert (exit_valid >= 0).all() and (exit_valid <= 100).all()


def test_gate_suppresses_when_r_low():
    """When R_t < r_thresh, Gate → 0 and CompositeScore suppressed."""
    from indicators.composite_refactor import compute_gate_refactor, compute_composite_score_refactor
    n = 50
    idx = pd.RangeIndex(n)
    comp = _synthetic_components(n, seed=2)
    comp["R_t"] = pd.Series(np.zeros(n) + 0.1, index=idx)  # below r_thresh 0.2
    comp["C_t"] = pd.Series(np.ones(n), index=idx)
    comp["L_t"] = pd.Series(np.ones(n), index=idx)
    gate = compute_gate_refactor(comp, r_thresh=0.2)
    assert (gate <= 0.01).all() or gate.max() < 0.2
    entry, _, _ = compute_composite_score_refactor(comp)
    # Entry should be low when gate is 0 (RawFavor = Opp * 0)
    assert entry.mean() < 60  # suppressed vs neutral 50


def test_composite_determinism():
    """Same components and config → same Entry/Exit (hash)."""
    import hashlib
    from indicators.composite_refactor import compute_composite_score_refactor
    comp = _synthetic_components(60, seed=3)
    e1, x1, _ = compute_composite_score_refactor(comp)
    e2, x2, _ = compute_composite_score_refactor(comp)
    assert hashlib.sha256(e1.values.tobytes()).hexdigest() == hashlib.sha256(e2.values.tobytes()).hexdigest()
    assert hashlib.sha256(x1.values.tobytes()).hexdigest() == hashlib.sha256(x2.values.tobytes()).hexdigest()


def test_g_pers_refactor_shape():
    from indicators.composite_refactor import g_pers_refactor
    H = np.array([0.3, 0.5, 0.7])
    g = g_pers_refactor(H, k_pers=6.0)
    assert g.shape == H.shape
    assert g[1] == pytest.approx(0.5, abs=1e-9)  # H=0.5 → g=0.5
    assert g[0] < 0.5 and g[2] > 0.5


def test_opportunity_uses_trimmed_mean():
    """Opportunity is deterministic and in [0,1]; with constant inputs it is constant."""
    from indicators.composite_refactor import compute_opportunity_refactor
    n = 40
    idx = pd.RangeIndex(n)
    # U_weighted = U_t * g_pers(H_t); g_pers(0.5)=0.5 so U_weighted=0.25 if U_t=0.5
    # Use U_t=1.0 so U_weighted=0.5 and all four inputs are 0.5
    comp = {
        "T_t": pd.Series(0.5 * np.ones(n), index=idx),
        "U_t": pd.Series(1.0 * np.ones(n), index=idx),
        "H_t": pd.Series(0.5 * np.ones(n), index=idx),
        "LDC_t": pd.Series(0.5 * np.ones(n), index=idx),
        "O_t": pd.Series(0.5 * np.ones(n), index=idx),
    }
    opp = compute_opportunity_refactor(comp, trim_frac=0.1, k_pers=6.0)
    assert (opp.dropna() >= 0).all() and (opp.dropna() <= 1).all()
    assert opp.nunique() == 1  # constant
