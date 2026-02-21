"""
TBL (Triple Barrier) manager and OU variance helper.

TBLManager(entry_price, tp, sl, tmax); on_tick(price, t) -> (exit_flag, reason).
estimate_ou_params(series), var_future(theta, sigma, T).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Optional


class TBLManager:
    """
    Triple barrier: upper (tp), lower (sl), time (tmax).
    on_tick(price, t) returns (exit_flag, reason) where reason in ("tp", "sl", "time") or "".
    """

    def __init__(
        self,
        entry_price: float,
        tp: float,
        sl: float,
        tmax: float,
        entry_time: Optional[float] = None,
    ) -> None:
        self.entry_price = float(entry_price)
        self.tp = float(tp)
        self.sl = float(sl)
        self.tmax = float(tmax)
        self._entry_time: Optional[float] = None
        self._entry_time_set = entry_time is not None
        if entry_time is not None:
            self._entry_time = float(entry_time)

    def on_tick(self, price: float, t: float) -> Tuple[bool, str]:
        """
        Check barriers at (price, t). Set entry_time on first call if not set in constructor.

        Returns
        -------
        (exit_flag, reason): reason is "tp", "sl", "time", or "".
        """
        if not self._entry_time_set:
            self._entry_time = float(t)
            self._entry_time_set = True
        entry_time = self._entry_time
        upper = self.entry_price + self.tp
        lower = self.entry_price - self.sl
        if price >= upper:
            return True, "tp"
        if price <= lower:
            return True, "sl"
        if t >= entry_time + self.tmax:
            return True, "time"
        return False, ""


def estimate_ou_params(series: pd.Series) -> Tuple[float, float, float]:
    """
    Estimate (theta, mu, sigma) for dX = θ(μ − X)dt + σ dW from discretized OU.

    Uses OLS on X_t - X_{t-1} = α + β X_{t-1} + ε; then θ = -β, μ = -α/β, σ = std(ε)*sqrt(2θ/(1-exp(-2θ*dt))).
    Assumes dt=1 if index is integer; else infers dt from index.
    """
    x = np.asarray(series, dtype=np.float64)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        raise ValueError("series must have at least 3 non-NaN points")
    y = np.diff(x)
    X_lag = x[:-1].reshape(-1, 1)
    # y = α + β x_{t-1}  =>  [1, x_{t-1}] @ [α, β]' = y
    ones = np.ones((len(X_lag), 1), dtype=np.float64)
    reg = np.hstack([ones, X_lag])
    coeffs, residuals, rank, _ = np.linalg.lstsq(reg, y, rcond=None)
    alpha, beta = coeffs[0], coeffs[1]
    theta = -float(beta)
    if theta <= 0:
        theta = 0.01
    # Discrete OU: X_t - X_{t-1} = θ μ dt - θ X_{t-1} dt + ε => α = θ μ, β = -θ => μ = α/θ
    mu = float(alpha) / theta if theta != 0 else float(np.mean(x))
    resid = y - reg @ coeffs
    sigma_resid = np.std(resid)
    if sigma_resid <= 0:
        sigma_resid = 1e-10
    dt = 1.0
    if hasattr(series, "index") and len(series.index) > 1:
        try:
            idx = series.dropna().index
            if hasattr(idx[0], "timestamp"):
                dt = (idx[1] - idx[0]).total_seconds() / 3600.0 if hasattr(idx[0], "total_seconds") else 1.0
            elif isinstance(series.index[0], (int, float)):
                dt = float(series.index[1] - series.index[0]) if len(series.index) > 1 else 1.0
        except Exception:
            dt = 1.0
    # Discrete: var(ε) ≈ σ² * (1 - exp(-2θ*dt)) / (2θ)  =>  σ² = var(ε) * 2θ / (1 - exp(-2θ*dt))
    factor = 1.0 - np.exp(-2 * theta * dt)
    if factor <= 0:
        factor = 1e-10
    sigma_sq = (sigma_resid ** 2) * (2 * theta) / factor
    sigma = float(np.sqrt(max(sigma_sq, 1e-20)))
    return theta, mu, sigma


def var_future(theta: float, sigma: float, T: float) -> float:
    """
    Var[X_T] for OU process: (σ²/(2θ)) * (1 - exp(-2θ T)).
    """
    if theta <= 0:
        return float("inf")
    return (sigma ** 2 / (2 * theta)) * (1.0 - np.exp(-2 * theta * T))
