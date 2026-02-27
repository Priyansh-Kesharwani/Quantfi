"""
Refactor-path weighting: IC-EWMA (primary) and Kalman/RLS (fallback).

WeightingFactory.create(method, config) returns a weighting instance.
Methods: ic_ewma, kalman_online (stub), regime_conditional (stub), online_hedge (stub).
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import numpy as np

from weights.ic_ewma import IC_EWMA_Weights
from weights.kalman import KalmanWeightsStub


class WeightingFactory:
    """Create weighting instances from config."""

    @staticmethod
    def create(method: str, config: Optional[Dict[str, Any]] = None) -> Any:
        """
        method: one of ic_ewma, kalman_online, regime_conditional, online_hedge.
        config: dict with method-specific keys (e.g. ic_window, alpha, lambda_shrink, max_weight_delta).
        """
        config = config or {}
        if method == "ic_ewma":
            return IC_EWMA_Weights(config)
        if method == "kalman_online":
            return KalmanWeightsStub(config)
        if method in ("regime_conditional", "online_hedge"):
            raise NotImplementedError(f"Weighting method {method!r} not implemented")
        raise ValueError(f"Unknown weighting method: {method!r}")
