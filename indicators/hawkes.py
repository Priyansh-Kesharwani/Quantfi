"""
Hawkes Intensity λ estimator — Phase A microstructure indicator.

Models self-exciting point processes via an exponential kernel:

    λ(t) = μ + Σ_{t_i < t}  α · e^{−β (t − t_i)}

Provides two estimation backends:
  1. `tick.HawkesExpKern` (if installed)
  2. Custom MLE fallback using scipy.optimize
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Any
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)

# ── Try importing tick ──────────────────────────────────────
_HAS_TICK = False
try:
    from tick.hawkes import HawkesExpKern  # type: ignore
    _HAS_TICK = True
except ImportError:
    logger.info("tick not installed — using custom MLE fallback for Hawkes estimation.")


# ── Custom MLE Fallback ────────────────────────────────────
def _hawkes_log_likelihood(
    params: np.ndarray,
    event_times: np.ndarray,
    T_end: float,
) -> float:
    """Negative log-likelihood of a univariate Hawkes process.

    params = [mu, alpha, beta]
    """
    mu, alpha, beta = params
    if mu <= 0 or alpha < 0 or beta <= 0 or alpha >= beta:
        return 1e12   # infeasible

    n = len(event_times)
    if n == 0:
        return mu * T_end

    # Recursive computation of Σ e^{-β(t_i - t_j)} for j < i
    A = np.zeros(n)
    for i in range(1, n):
        A[i] = np.exp(-beta * (event_times[i] - event_times[i - 1])) * (1.0 + A[i - 1])

    # log-likelihood components
    lambda_vals = mu + alpha * A
    lambda_vals = np.maximum(lambda_vals, 1e-30)

    ll = np.sum(np.log(lambda_vals))
    ll -= mu * T_end
    ll -= (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T_end - event_times)))

    return -ll   # negate for minimisation


def _fit_hawkes_mle(
    event_times: np.ndarray,
    T_end: float,
    mu_init: float = 0.1,
    alpha_init: float = 0.5,
    beta_init: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> Tuple[float, float, float]:
    """Fit μ, α, β via L-BFGS-B on the negative log-likelihood."""
    x0 = np.array([mu_init, alpha_init, beta_init])
    bounds = [(1e-8, None), (1e-8, None), (1e-8, None)]

    res = minimize(
        _hawkes_log_likelihood,
        x0,
        args=(event_times, T_end),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter, "ftol": tol},
    )

    mu, alpha, beta = res.x
    # Ensure stationarity: α < β
    if alpha >= beta:
        alpha = beta * 0.99
        logger.warning("Hawkes MLE: α ≥ β, clamped to 0.99·β for stationarity.")

    return float(mu), float(alpha), float(beta)


# ── Intensity computation ──────────────────────────────────
def _compute_intensity(
    event_times: np.ndarray,
    timestamps: np.ndarray,
    mu: float,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Evaluate λ(t) on a regular grid of *timestamps*.

    λ(t) = μ + Σ_{t_i < t} α · e^{−β(t − t_i)}
    """
    n_grid = len(timestamps)
    lam = np.full(n_grid, mu, dtype=np.float64)

    for k, t in enumerate(timestamps):
        past = event_times[event_times < t]
        if len(past) > 0:
            lam[k] += alpha * np.sum(np.exp(-beta * (t - past)))

    return lam


# ── Public API ─────────────────────────────────────────────
def estimate_hawkes(
    events: Dict[str, np.ndarray],
    timestamps: np.ndarray,
    dt: float = 1.0,
    decay: float = 1.0,
    mu_init: float = 0.1,
    alpha_init: float = 0.5,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Estimate Hawkes intensity λ(t) on a regular timestamp grid.

    Parameters
    ----------
    events : dict
        Mapping of event-type names → 1-D arrays of event arrival times.
        For univariate Hawkes, pass e.g. ``{"trades": np.array([...])}``.
    timestamps : np.ndarray
        1-D array of evaluation times (regular grid, spacing *dt*).
    dt : float
        Grid spacing (default 1.0 — one bar).
    decay : float
        β (exponential-kernel decay).  Used as initial value for MLE.
    mu_init : float
        Initial baseline intensity for MLE.
    alpha_init : float
        Initial excitation parameter for MLE.
    max_iter : int
        Maximum MLE iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    intensity : pd.Series
        λ(t) evaluated at each *timestamps* entry.
    meta : dict
        Fitted parameters and diagnostic info.
    """
    # Flatten all events into a single sorted array (univariate aggregation)
    all_events = np.sort(np.concatenate(list(events.values())))
    T_end = float(timestamps[-1]) if len(timestamps) > 0 else 1.0

    meta: Dict[str, Any] = {"backend": None, "mu": None, "alpha": None, "beta": None}

    if _HAS_TICK and len(all_events) > 5:
        # ── tick backend ──────────────────────────
        try:
            learner = HawkesExpKern(decay, max_iter=max_iter, tol=tol, verbose=False)
            learner.fit([all_events.tolist()])
            mu_hat = float(learner.baseline[0])
            alpha_hat = float(learner.adjacency[0, 0])
            beta_hat = decay
            meta["backend"] = "tick"
        except Exception as e:
            logger.warning(f"tick backend failed ({e}), falling back to custom MLE.")
            mu_hat, alpha_hat, beta_hat = _fit_hawkes_mle(
                all_events, T_end, mu_init, alpha_init, decay, max_iter, tol
            )
            meta["backend"] = "custom_mle"
    else:
        # ── custom MLE fallback ───────────────────
        mu_hat, alpha_hat, beta_hat = _fit_hawkes_mle(
            all_events, T_end, mu_init, alpha_init, decay, max_iter, tol
        )
        meta["backend"] = "custom_mle"

    meta["mu"] = mu_hat
    meta["alpha"] = alpha_hat
    meta["beta"] = beta_hat

    lam = _compute_intensity(all_events, timestamps, mu_hat, alpha_hat, beta_hat)

    intensity = pd.Series(lam, index=pd.RangeIndex(len(timestamps)), name="hawkes_lambda")
    meta["n_events"] = len(all_events)
    meta["T_end"] = T_end
    meta["mean_lambda"] = float(np.mean(lam))

    return intensity, meta


def hawkes_lambda_decay(
    events: Dict[str, np.ndarray],
    timestamps: np.ndarray,
    decay: float = 1.0,
    min_obs: int = 50,
    norm_k: float = 1.0,
    **kwargs,
) -> pd.Series:
    """Convenience: normalised λ-decay signal for the Exit Score.

    Uses the expanding-ECDF → inverse-normal → sigmoid pipeline with
    *inverted* polarity so that high values indicate event intensity
    is falling (supportive of exit).

    Parameters
    ----------
    events, timestamps, decay : see ``estimate_hawkes``.
    min_obs : int
        Warm-up observations for expanding ECDF normalisation.
    norm_k : float
        Sigmoid steepness.
    **kwargs : forwarded to ``estimate_hawkes``.
    """
    from indicators.normalization import expanding_ecdf_sigmoid

    intensity, _meta = estimate_hawkes(events, timestamps, decay=decay, **kwargs)

    # Normalise via expanding ECDF-sigmoid with inverted polarity
    # polarity=-1 → low intensity maps to high score (exit signal)
    norm = expanding_ecdf_sigmoid(
        intensity, k=norm_k, polarity=-1, min_obs=min_obs,
    )
    norm.name = "hawkes_decay"

    return norm
