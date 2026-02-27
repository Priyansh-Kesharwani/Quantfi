"""
Lorentzian Distance Classifier (LDC).

Distance: d_L(x,y) = sum_i log(1 + ((x_i - y_i)/gamma_i)^2).
Score: s(x) = 1 / (1 + exp(kappa * (d_bull(x) - d_bear(x)))).
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional


def lorentzian_distance(
    x: np.ndarray,
    y: np.ndarray,
    gamma: Optional[np.ndarray] = None,
) -> float:
    """Lorentzian distance: sum_i log(1 + ((x_i - y_i)/gamma_i)^2)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError("x and y must have same shape")
    if gamma is None:
        gamma = np.ones_like(x)
    else:
        gamma = np.asarray(gamma, dtype=np.float64).ravel()
        if gamma.shape != x.shape:
            raise ValueError("gamma must match x shape")
    diff = (x - y) / np.maximum(gamma, 1e-12)
    return float(np.sum(np.log(1.0 + diff * diff)))


def build_templates_from_labels(
    features: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Build bull/bear templates as class means. labels: 1 = bull, 0 = bear."""
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.intp).ravel()
    if len(labels) != len(features):
        raise ValueError("labels and features length mismatch")
    bull_mask = labels == 1
    bear_mask = labels == 0
    if not np.any(bull_mask) or not np.any(bear_mask):
        raise ValueError("Need both bull (1) and bear (0) labels")
    return {
        "bull": np.mean(features[bull_mask], axis=0),
        "bear": np.mean(features[bear_mask], axis=0),
    }


class LDC:
    """Lorentzian Distance Classifier: score = sigmoid(kappa * (d_bear - d_bull))."""

    def __init__(self, kappa: float = 1.0):
        self.kappa = float(kappa)
        self._templates: Optional[Dict[str, np.ndarray]] = None

    def fit(self, templates: Dict[str, np.ndarray]) -> "LDC":
        if "bull" not in templates or "bear" not in templates:
            raise ValueError("templates must contain 'bull' and 'bear' keys")
        self._templates = {
            "bull": np.asarray(templates["bull"], dtype=np.float64).ravel(),
            "bear": np.asarray(templates["bear"], dtype=np.float64).ravel(),
        }
        if self._templates["bull"].shape != self._templates["bear"].shape:
            raise ValueError("bull and bear templates must have same shape")
        return self

    def score(self, x: np.ndarray) -> float:
        if self._templates is None:
            raise RuntimeError("LDC must be fitted before score()")
        x = np.asarray(x, dtype=np.float64).ravel()
        d_bull = lorentzian_distance(x, self._templates["bull"])
        d_bear = lorentzian_distance(x, self._templates["bear"])
        # s = 1/(1+exp(kappa*(d_bull - d_bear))): closer to bull -> d_bull small -> s > 0.5
        logit = self.kappa * (d_bear - d_bull)
        logit = np.clip(logit, -500, 500)
        return float(1.0 / (1.0 + np.exp(-logit)))

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.array([self.score(X[i]) for i in range(len(X))], dtype=np.float64)

    def to_dict(self) -> Dict[str, Any]:
        if self._templates is None:
            raise RuntimeError("LDC must be fitted before to_dict()")
        return {
            "kappa": self.kappa,
            "bull": self._templates["bull"].tolist(),
            "bear": self._templates["bear"].tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LDC":
        ldc = cls(kappa=d["kappa"])
        ldc._templates = {
            "bull": np.array(d["bull"], dtype=np.float64),
            "bear": np.array(d["bear"], dtype=np.float64),
        }
        return ldc
