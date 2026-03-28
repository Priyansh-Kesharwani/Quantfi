"""
Tests for refactor-path weighting (IC-EWMA, Kalman stub).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def test_weighting_factory_create_ic_ewma():
    from weights import WeightingFactory
    w = WeightingFactory.create("ic_ewma", {"alpha": 5.0, "lambda_shrink": 0.2})
    assert w is not None
    assert hasattr(w, "update") and hasattr(w, "weights")

def test_weighting_factory_create_kalman_stub():
    from weights import WeightingFactory
    w = WeightingFactory.create("kalman_online", {})
    assert w is not None
    w_arr, meta = w.update(np.random.randn(50, 3), np.random.randn(50))
    assert w_arr.shape == (3,)
    assert np.isclose(w_arr.sum(), 1.0)
    assert (w_arr >= 0).all()

def test_ic_ewma_weights_simplex():
    from weights import WeightingFactory
    np.random.seed(42)
    T, n = 100, 4
    cr = np.random.randn(T, n).cumsum(axis=0) * 0.01
    fwd = np.random.randn(T).cumsum() * 0.01
    w = WeightingFactory.create("ic_ewma", {"ic_window": 50, "ic_forward_horizon": 5})
    weights, meta = w.update(cr, fwd)
    assert weights.shape == (n,)
    assert np.isclose(weights.sum(), 1.0)
    assert (weights >= 0).all()
    assert "ewma_ic" in meta

def test_ic_ewma_higher_ic_higher_weight():
    """Component with higher IC should get higher weight; weights are simplex."""
    from engine.weights.ic_ewma import IC_EWMA_Weights
    np.random.seed(123)
    T = 80
    fwd = np.random.randn(T) * 0.01
    c0 = fwd + np.roll(fwd, -1) * 0.5
    c1 = np.random.randn(T) * 0.01
    cr = np.column_stack([c0, c1])
    config = {"ic_window": 60, "ic_forward_horizon": 5, "alpha": 3.0, "lambda_shrink": 0.1, "max_weight_delta": 0.5}
    w_engine = IC_EWMA_Weights(config)
    weights, meta = w_engine.update(cr, fwd)
    assert weights.shape == (2,)
    assert np.isclose(weights.sum(), 1.0)
    assert (weights >= 0).all()
    assert weights[0] > weights[1]

def test_ic_ewma_turnover_cap():
    """With max_weight_delta, weights evolve; output remains simplex."""
    from engine.weights.ic_ewma import IC_EWMA_Weights
    np.random.seed(44)
    T, n = 120, 3
    cr = np.random.randn(T, n).cumsum(axis=0) * 0.01
    fwd = np.random.randn(T).cumsum() * 0.01
    delta_max = 0.2
    w_engine = IC_EWMA_Weights({"max_weight_delta": delta_max, "ic_window": 40, "ic_forward_horizon": 5})
    for t in range(50, T, 10):
        w, _ = w_engine.update(cr[: t], fwd[: t])
        assert np.isclose(w.sum(), 1.0)
        assert (w >= 0).all()

def test_weighting_factory_unknown_method():
    from weights import WeightingFactory
    with pytest.raises(ValueError):
        WeightingFactory.create("unknown_method")
