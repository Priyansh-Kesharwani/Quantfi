"""
Kalman/RLS weighting stub for refactor path.

Fallback method; not fully implemented. Returns equal weights.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional, Tuple

class KalmanWeightsStub:
    """Stub: returns equal weights. Full Kalman/RLS to be added later."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self._n_components: Optional[int] = None
        self._weights: Optional[np.ndarray] = None

    def update(
        self,
        component_returns: np.ndarray,
        forward_returns: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Return equal weights (simplex)."""
        cr = np.asarray(component_returns, dtype=np.float64)
        if cr.ndim == 1:
            cr = cr.reshape(-1, 1)
        n = cr.shape[1]
        if self._n_components is None:
            self._n_components = n
            self._weights = np.ones(n) / n
        w = self._weights.copy()
        meta = {"raw_ic": [], "ewma_ic": [], "prior_shrink": 0.0, "method": "kalman_stub"}
        return w, meta

    def weights(self) -> np.ndarray:
        if self._weights is None:
            return np.ones(1)
        return self._weights.copy()
