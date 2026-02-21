"""
Unit tests for bot.bayes_online: BayesOnline and BayesHierarchical.

Synthetic linear model; convergence and determinism.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bot.bayes_online import BayesOnline, BayesHierarchical


def test_bayes_online_constructor():
    """BayesOnline(dim, lambda_, sigma2) accepts valid args."""
    b = BayesOnline(3, 0.95, 1.0)
    assert b.dim == 3
    assert b.predict(np.zeros(3)) == 0.0


def test_bayes_online_rejects_invalid_lambda():
    """lambda_ must be in (0, 1)."""
    with pytest.raises(ValueError, match="lambda"):
        BayesOnline(2, 1.0, 1.0)
    with pytest.raises(ValueError, match="lambda"):
        BayesOnline(2, 0.0, 1.0)


def test_bayes_online_update_predict():
    """Update then predict returns a float."""
    np.random.seed(42)
    b = BayesOnline(2, 0.98, 0.5)
    b.update(np.array([1.0, 0.5]), 2.0)
    p = b.predict(np.array([1.0, 0.5]))
    assert isinstance(p, float)


def test_bayes_online_convergence_on_linear_model():
    """On y = β'x + noise, predictions approach true β'x after many updates."""
    np.random.seed(43)
    dim = 4
    beta_true = np.array([0.5, -0.3, 0.8, 0.1])
    b = BayesOnline(dim, 0.99, 0.1)
    for _ in range(200):
        x = np.random.randn(dim)
        y = float(np.dot(beta_true, x)) + np.random.randn() * 0.3
        b.update(x, y)
    x_test = np.random.randn(dim)
    pred = b.predict(x_test)
    true_val = np.dot(beta_true, x_test)
    assert abs(pred - true_val) < 1.0


def test_bayes_online_deterministic():
    """Same (x,y) sequence and params produce same predictions."""
    np.random.seed(44)
    dim = 3
    b1 = BayesOnline(dim, 0.97, 0.5)
    b2 = BayesOnline(dim, 0.97, 0.5)
    for _ in range(50):
        x = np.random.randn(dim)
        y = float(np.random.randn())
        b1.update(x, y)
        b2.update(x, y)
    x = np.random.randn(dim)
    assert b1.predict(x) == b2.predict(x)


def test_bayes_online_wrong_dim_raises():
    """update/predict with wrong x length raise."""
    b = BayesOnline(2, 0.95, 1.0)
    with pytest.raises(ValueError, match="length"):
        b.update(np.array([1.0, 2.0, 3.0]), 1.0)
    with pytest.raises(ValueError, match="length"):
        b.predict(np.array([1.0]))


def test_bayes_hierarchical_update_for_regime():
    """BayesHierarchical.update_for_regime(regime_id, x, y) and predict(x, regime_id)."""
    np.random.seed(45)
    h = BayesHierarchical(2, 0.96, 0.4, eta=0.1)
    h.update_for_regime(0, np.array([1.0, 0.0]), 1.0)
    h.update_for_regime(1, np.array([0.0, 1.0]), -0.5)
    p0 = h.predict(np.array([1.0, 0.0]), regime_id=0)
    p1 = h.predict(np.array([0.0, 1.0]), regime_id=1)
    assert isinstance(p0, float)
    assert isinstance(p1, float)
    p_global = h.predict(np.array([1.0, 0.0]))
    assert isinstance(p_global, float)


def test_bayes_hierarchical_two_regimes_deterministic():
    """Same sequence to two regimes gives same predictions."""
    np.random.seed(46)
    seq = [(0, np.array([1.0, 0.2]), 0.5), (1, np.array([0.2, 1.0]), -0.3)] * 20
    h1 = BayesHierarchical(2, 0.95, 0.5, eta=0.15)
    h2 = BayesHierarchical(2, 0.95, 0.5, eta=0.15)
    for r, x, y in seq:
        h1.update_for_regime(r, x, y)
        h2.update_for_regime(r, x, y)
    assert h1.predict(np.array([1.0, 0.0]), 0) == h2.predict(np.array([1.0, 0.0]), 0)
