"""
Online Bayesian linear regression with forgetting factor λ.

BayesOnline: update(x, y), predict(x).
BayesHierarchical: update_for_regime(regime_id, x, y) with shrinkage toward global.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional


class BayesOnline:
    """
    Bayesian linear regression with exponential forgetting (λ ∈ (0, 1)).

    Model: y_t = β' x_t + ε_t, ε_t ~ N(0, σ²).
    Prior: β ~ N(μ_t, Σ_t). Update:
      Σ_{t+1}^{-1} = (1/λ) Σ_t^{-1} + (1/σ²) x x'
      μ_{t+1} = Σ_{t+1} ( (1/λ) Σ_t^{-1} μ_t + (1/σ²) x y )
    """

    def __init__(self, dim: int, lambda_: float, sigma2: float) -> None:
        if not 0 < lambda_ < 1:
            raise ValueError("lambda_ must be in (0, 1)")
        if sigma2 <= 0:
            raise ValueError("sigma2 must be positive")
        self.dim = int(dim)
        self.lambda_ = float(lambda_)
        self.sigma2 = float(sigma2)
        self._mu = np.zeros(self.dim, dtype=np.float64)
        self._Sigma_inv = np.eye(self.dim, dtype=np.float64) * 1e-6

    def update(self, x: np.ndarray, y: float) -> None:
        """Update posterior with observation (x, y). x shape (dim,) or (1, dim)."""
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.shape[0] != self.dim:
            raise ValueError(f"x must have length {self.dim}, got {x.shape[0]}")
        # b = (1/λ) Σ_t^{-1} μ_t + (1/σ²) x y  (use current Sigma_inv)
        b = (1.0 / self.lambda_) * (self._Sigma_inv @ self._mu) + (1.0 / self.sigma2) * x * y
        # Σ_{t+1}^{-1} = (1/λ) Σ_t^{-1} + (1/σ²) x x'
        self._Sigma_inv = (1.0 / self.lambda_) * self._Sigma_inv + (1.0 / self.sigma2) * np.outer(x, x)
        # μ_{t+1} = Σ_{t+1} b
        self._mu = np.linalg.solve(self._Sigma_inv, b)

    def predict(self, x: np.ndarray) -> float:
        """Predict E[y] = μ' x."""
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.shape[0] != self.dim:
            raise ValueError(f"x must have length {self.dim}, got {x.shape[0]}")
        return float(np.dot(self._mu, x))


class BayesHierarchical:
    """
    Hierarchical Bayesian online: global prior + per-regime models with shrinkage.
    update_for_regime(regime_id, x, y) updates the regime model and optionally shrinks toward global.
    """

    def __init__(
        self,
        dim: int,
        lambda_: float,
        sigma2: float,
        eta: float = 0.1,
    ) -> None:
        self.dim = dim
        self.eta = float(eta)
        self._global = BayesOnline(dim, lambda_, sigma2)
        self._regimes: Dict[int, BayesOnline] = {}

    def _get_regime(self, regime_id: int) -> BayesOnline:
        if regime_id not in self._regimes:
            self._regimes[regime_id] = BayesOnline(self.dim, self._global.lambda_, self._global.sigma2)
        return self._regimes[regime_id]

    def update_for_regime(self, regime_id: int, x: np.ndarray, y: float) -> None:
        """Update global with (x,y), then update regime model; optionally shrink regime toward global."""
        x = np.asarray(x, dtype=np.float64).ravel()
        self._global.update(x, y)
        r = self._get_regime(regime_id)
        r.update(x, y)
        # Shrink regime mean toward global: μ_r := (1-η) μ_r + η μ_global
        self._regimes[regime_id]._mu = (1.0 - self.eta) * r._mu + self.eta * self._global._mu

    def predict(self, x: np.ndarray, regime_id: Optional[int] = None) -> float:
        """Predict using regime model if regime_id given, else global."""
        x = np.asarray(x, dtype=np.float64).ravel()
        if regime_id is not None and regime_id in self._regimes:
            return self._regimes[regime_id].predict(x)
        return self._global.predict(x)
