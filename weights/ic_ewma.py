from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional, Tuple

MODE_OFFLINE = "offline"
MODE_LIVE = "live"
LIVE_FORWARD_MSG = "IC_EWMA_Weights: mode='live' forbids use of forward returns; use mode='offline' for evaluation only."

def _spearman_ic(x: np.ndarray, y: np.ndarray) -> float:
    """Sample Spearman correlation (IC)."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if np.sum(mask) < 3:
        return np.nan
    from scipy.stats import spearmanr
    r, _ = spearmanr(x[mask], y[mask])
    return float(r) if not np.isnan(r) else np.nan

class IC_EWMA_Weights:
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        ic_window: int = 252,
        ic_forward_horizon: int = 20,
        alpha: float = 5.0,
        lambda_shrink: float = 0.2,
        max_weight_delta: float = 0.1,
        eps: float = 1e-8,
        mode: str = MODE_OFFLINE,
    ):
        config = config or {}
        self.ic_window = config.get("ic_window", ic_window)
        self.ic_forward_horizon = config.get("ic_forward_horizon", ic_forward_horizon)
        self.alpha = config.get("alpha", alpha)
        self.lambda_shrink = config.get("lambda_shrink", lambda_shrink)
        self.max_weight_delta = config.get("max_weight_delta", max_weight_delta)
        self.eps = eps
        self.mode = config.get("mode", mode)
        if self.mode not in (MODE_OFFLINE, MODE_LIVE):
            raise ValueError("mode must be 'offline' or 'live'")
        self._n_components: Optional[int] = None
        self._prev_weights: Optional[np.ndarray] = None
        self._ewma_ic: Optional[np.ndarray] = None
        self._t: int = 0

    def update(
        self,
        component_returns: np.ndarray,
        forward_returns: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self.mode == MODE_LIVE:
            raise RuntimeError(LIVE_FORWARD_MSG)
        forward_returns = np.asarray(forward_returns).ravel()
        cr = np.asarray(component_returns, dtype=np.float64)
        if cr.ndim == 1:
            cr = cr.reshape(-1, 1)
        T, n = cr.shape
        if self._n_components is None:
            self._n_components = n
            self._prev_weights = np.ones(n) / n
            self._ewma_ic = np.zeros(n)
        assert n == self._n_components

        start = max(0, T - self.ic_window)
        fwd = forward_returns[start:T]
        if len(fwd) < self.ic_forward_horizon:
            w = self._prev_weights.copy()
            meta = {"raw_ic": [], "ewma_ic": self._ewma_ic.tolist(), "prior_shrink": self.lambda_shrink}
            return w, meta

        n_vals = T - start - self.ic_forward_horizon
        horizon = min(self.ic_forward_horizon, max(1, len(forward_returns) - start - 1))
        if horizon < 1:
            horizon = 1
        n_vals = T - start - horizon
        if n_vals < 3:
            w = self._prev_weights.copy()
            meta = {"raw_ic": [0.0] * n, "ewma_ic": self._ewma_ic.tolist(), "prior_shrink": self.lambda_shrink}
            return w, meta
        raw_ic = np.zeros(n)
        for i in range(n):
            x = cr[start : start + n_vals, i]
            y = np.array([
                np.mean(forward_returns[start + j : start + j + horizon])
                for j in range(n_vals)
            ], dtype=np.float64)
            raw_ic[i] = _spearman_ic(x, y)
        raw_ic = np.nan_to_num(raw_ic, nan=0.0)

        if self._t == 0:
            self._ewma_ic = raw_ic.copy()
        else:
            beta = 0.1
            self._ewma_ic = (1 - beta) * self._ewma_ic + beta * raw_ic
        self._t += 1

        mu_ic = np.mean(self._ewma_ic)
        sigma_ic = np.std(self._ewma_ic) + self.eps
        z = (self._ewma_ic - mu_ic) / sigma_ic
        w_tilde = np.exp(self.alpha * z)
        w_tilde = np.maximum(w_tilde, 1e-12)
        w_shrunk = (1 - self.lambda_shrink) * w_tilde + self.lambda_shrink * (1.0 / n)
        w = w_shrunk / np.sum(w_shrunk)
        delta = np.clip(w - self._prev_weights, -self.max_weight_delta, self.max_weight_delta)
        w = self._prev_weights + delta
        w = np.maximum(w, 0)
        w = w / np.sum(w)
        self._prev_weights = w.copy()

        meta = {
            "raw_ic": raw_ic.tolist(),
            "ewma_ic": self._ewma_ic.tolist(),
            "prior_shrink": self.lambda_shrink,
        }
        return w, meta

    def weights(self) -> np.ndarray:
        """Return current weights (simplex)."""
        if self._prev_weights is None:
            return np.ones(self._n_components or 1) / (self._n_components or 1)
        return self._prev_weights.copy()
