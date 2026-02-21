"""
Phase B+3 — Hawkes Process Simulator for Edge Testing & Execution Stress.

Generates controlled synthetic event streams with specified:
  - Baseline intensity μ
  - Excitation parameter α
  - Decay rate β
  - Inter-event characteristics

Phase 3 additions:
  - Synthetic LOB snapshot generation from Hawkes event streams
  - Synthetic trade tick generation for OFI stress testing
  - Queue dynamics simulation

Provides ground-truth λ(t) for comparison against estimated λ(t).

Stress-test cases:
  1. Low μ, high α  → bursty regime
  2. High μ, low α  → near-Poisson
  3. α ≈ β          → explosive branching ratio edge
  4. Low event count → MLE stability test
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def simulate_hawkes_events(
    mu: float,
    alpha: float,
    beta: float,
    T: float,
    seed: int = 0,
) -> np.ndarray:
    """Simulate Hawkes process event times via Ogata's thinning algorithm.

    Parameters
    ----------
    mu : float
        Baseline intensity (background rate).
    alpha : float
        Excitation parameter (jump per event).
    beta : float
        Exponential decay rate.
    T : float
        Total observation interval [0, T].
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    np.ndarray
        Sorted array of event timestamps in [0, T].
    """
    rng = np.random.RandomState(seed)
    events: List[float] = []
    t = 0.0
    lam_star = mu  # initial intensity upper bound

    while t < T:
        # Draw next candidate inter-arrival
        u = rng.rand()
        if u <= 0:
            u = 1e-15
        w = -np.log(u) / lam_star
        t += w
        if t >= T:
            break

        # Compute actual intensity at t
        lam_t = mu
        for s in events:
            lam_t += alpha * np.exp(-beta * (t - s))

        # Accept/reject
        d = rng.rand()
        if d <= lam_t / lam_star:
            events.append(t)
            lam_star = lam_t + alpha  # update upper bound
        else:
            lam_star = max(lam_t, mu)  # tighten bound

    return np.array(events)


def ground_truth_intensity(
    events: np.ndarray,
    mu: float,
    alpha: float,
    beta: float,
    grid: np.ndarray,
) -> np.ndarray:
    """Compute exact λ(t) on a time grid given known parameters.

    λ(t) = μ + Σ_{t_i < t} α · exp(-β · (t - t_i))

    Parameters
    ----------
    events : np.ndarray
        Event timestamps.
    mu : float
        Baseline intensity.
    alpha : float
        Excitation.
    beta : float
        Decay.
    grid : np.ndarray
        Evaluation times.

    Returns
    -------
    np.ndarray
        True λ(t) at each grid point.
    """
    n_grid = len(grid)
    lam = np.full(n_grid, mu, dtype=np.float64)

    # Use recursive computation for efficiency
    # Sort events (should already be sorted)
    events_sorted = np.sort(events)

    for k, t in enumerate(grid):
        past = events_sorted[events_sorted < t]
        if len(past) > 0:
            lam[k] += alpha * np.sum(np.exp(-beta * (t - past)))

    return lam


def branching_ratio(alpha: float, beta: float) -> float:
    """Compute Hawkes branching ratio η = α/β.

    - η < 1: stationary (sub-critical)
    - η = 1: critical
    - η > 1: explosive (super-critical)

    Returns
    -------
    float
        Branching ratio.
    """
    if beta <= 0:
        return np.inf
    return alpha / beta


def expected_event_rate(mu: float, alpha: float, beta: float) -> float:
    """Expected stationary event rate: μ / (1 - α/β).

    Only valid when α < β (stationary regime).

    Returns
    -------
    float
        Expected event rate, or np.inf if not stationary.
    """
    eta = branching_ratio(alpha, beta)
    if eta >= 1.0:
        return np.inf
    return mu / (1.0 - eta)


def intensity_rmse(
    estimated: np.ndarray,
    truth: np.ndarray,
) -> float:
    """Root Mean Squared Error between estimated and true intensity.

    Parameters
    ----------
    estimated : np.ndarray
        Estimated λ(t) values.
    truth : np.ndarray
        Ground-truth λ(t) values.

    Returns
    -------
    float
        RMSE.
    """
    return float(np.sqrt(np.mean((estimated - truth) ** 2)))


def relative_rmse(
    estimated: np.ndarray,
    truth: np.ndarray,
) -> float:
    """RMSE normalised by mean of truth.

    Returns
    -------
    float
        Relative RMSE (fraction), e.g. 0.10 = 10%.
    """
    mean_truth = np.mean(truth)
    if mean_truth == 0:
        return np.inf
    return intensity_rmse(estimated, truth) / mean_truth


# ====================================================================
# Regime Simulation Scenarios
# ====================================================================

def simulate_regime(
    regime_name: str,
    mu: float,
    alpha: float,
    beta: float,
    T: float,
    seed: int = 0,
    grid_resolution: float = 0.1,
) -> Dict[str, Any]:
    """Simulate a single Hawkes regime and compute ground-truth λ(t).

    Parameters
    ----------
    regime_name : str
        Descriptive name for this regime (e.g. "bursty").
    mu, alpha, beta : float
        Hawkes process parameters.
    T : float
        Observation interval.
    seed : int
        RNG seed.
    grid_resolution : float
        Time step for the evaluation grid.

    Returns
    -------
    dict with keys: regime_name, events, grid, true_intensity,
         mu, alpha, beta, branching_ratio, expected_rate, n_events.
    """
    events = simulate_hawkes_events(mu, alpha, beta, T, seed=seed)
    grid = np.arange(0, T, grid_resolution)
    true_lam = ground_truth_intensity(events, mu, alpha, beta, grid)

    return {
        "regime_name": regime_name,
        "events": events,
        "grid": grid,
        "true_intensity": true_lam,
        "mu": mu,
        "alpha": alpha,
        "beta": beta,
        "branching_ratio": branching_ratio(alpha, beta),
        "expected_rate": expected_event_rate(mu, alpha, beta),
        "n_events": len(events),
        "T": T,
        "seed": seed,
    }


def run_all_regimes(
    regimes: List[Dict[str, Any]],
    base_seed: int = 0,
    grid_resolution: float = 0.1,
) -> List[Dict[str, Any]]:
    """Simulate all configured Hawkes regimes.

    Parameters
    ----------
    regimes : list of dict
        Each dict must have: name, mu, alpha, beta, T.
    base_seed : int
        Base RNG seed (each regime gets base_seed + i).
    grid_resolution : float
        Time step for evaluation grids.

    Returns
    -------
    list of regime result dicts.
    """
    results = []
    for i, reg in enumerate(regimes):
        seed = base_seed + i
        result = simulate_regime(
            regime_name=reg["name"],
            mu=reg["mu"],
            alpha=reg["alpha"],
            beta=reg["beta"],
            T=reg["T"],
            seed=seed,
            grid_resolution=grid_resolution,
        )
        logger.info(
            f"  Regime '{reg['name']}': {result['n_events']} events, "
            f"η={result['branching_ratio']:.3f}, "
            f"mean λ={np.mean(result['true_intensity']):.3f}"
        )
        results.append(result)

    return results


def validate_estimation(
    estimated_intensity: np.ndarray,
    regime_result: Dict[str, Any],
    rmse_threshold: float = 0.10,
) -> Dict[str, Any]:
    """Validate estimated λ(t) against ground truth from a regime simulation.

    Parameters
    ----------
    estimated_intensity : np.ndarray
        Estimated λ(t) on the regime's grid.
    regime_result : dict
        Output from simulate_regime.
    rmse_threshold : float
        Maximum allowed relative RMSE.

    Returns
    -------
    dict with validation results.
    """
    truth = regime_result["true_intensity"]

    # Align lengths (estimation might be on different grid)
    min_len = min(len(estimated_intensity), len(truth))
    est = estimated_intensity[:min_len]
    tru = truth[:min_len]

    rmse = intensity_rmse(est, tru)
    rel_rmse = relative_rmse(est, tru)
    passed = rel_rmse <= rmse_threshold

    return {
        "regime_name": regime_result["regime_name"],
        "rmse": rmse,
        "relative_rmse": rel_rmse,
        "rmse_threshold": rmse_threshold,
        "passed": passed,
        "n_events": regime_result["n_events"],
        "branching_ratio": regime_result["branching_ratio"],
        "mean_estimated": float(np.mean(est)),
        "mean_truth": float(np.mean(tru)),
    }


# ====================================================================
# Phase 3 — Synthetic LOB & Trade Tick Generation
# ====================================================================

def generate_synthetic_lob(
    events: np.ndarray,
    grid: np.ndarray,
    base_mid: float = 100.0,
    tick_size: float = 0.01,
    depth_levels: int = 5,
    base_depth: float = 100.0,
    volatility: float = 0.001,
    seed: int = 0,
) -> Dict[str, Any]:
    """Generate synthetic LOB snapshots from Hawkes event stream.

    Creates bid/ask ladders at each grid point, with depth modulated
    by the local event rate (more events → thinner book near mid).

    Parameters
    ----------
    events : np.ndarray
        Hawkes event timestamps.
    grid : np.ndarray
        Time grid for LOB snapshots.
    base_mid : float
        Starting mid-price.
    tick_size : float
        Price tick size.
    depth_levels : int
        Number of LOB levels on each side.
    base_depth : float
        Base quantity at each level.
    volatility : float
        Per-step price volatility.
    seed : int
        RNG seed.

    Returns
    -------
    dict with keys: grid, mid_prices, bid_prices, ask_prices,
         bid_depths, ask_depths (each arrays or nested lists).
    """
    rng = np.random.RandomState(seed)
    n = len(grid)

    # Simulate mid-price random walk
    returns = rng.normal(0, volatility, n)
    mid_prices = base_mid * np.exp(np.cumsum(returns))

    # Count events in windows around each grid point
    event_counts = np.zeros(n)
    for k, t in enumerate(grid):
        window = 1.0
        event_counts[k] = np.sum(
            (events >= t - window) & (events < t + window)
        )

    # Normalised activity (higher activity → thinner book)
    max_count = max(event_counts.max(), 1)
    activity = event_counts / max_count  # [0, 1]

    bid_prices_all = []
    ask_prices_all = []
    bid_depths_all = []
    ask_depths_all = []

    for k in range(n):
        mid = mid_prices[k]
        spread_mult = 1.0 + activity[k] * 2.0  # wider spread when active

        bids = []
        asks = []
        b_depths = []
        a_depths = []
        for lev in range(depth_levels):
            offset = tick_size * (lev + 1) * spread_mult
            bids.append(mid - offset)
            asks.append(mid + offset)

            # Depth decreases closer to mid, modulated by activity
            depth = base_depth * (1.0 - 0.5 * activity[k]) / (lev + 1)
            depth = max(depth + rng.normal(0, depth * 0.1), 1.0)
            b_depths.append(depth)
            a_depths.append(depth)

        bid_prices_all.append(bids)
        ask_prices_all.append(asks)
        bid_depths_all.append(b_depths)
        ask_depths_all.append(a_depths)

    return {
        "grid": grid,
        "mid_prices": mid_prices,
        "bid_prices": bid_prices_all,
        "ask_prices": ask_prices_all,
        "bid_depths": bid_depths_all,
        "ask_depths": ask_depths_all,
        "event_counts": event_counts,
        "n_levels": depth_levels,
    }


def generate_synthetic_trades(
    events: np.ndarray,
    base_mid: float = 100.0,
    volatility: float = 0.001,
    buy_prob: float = 0.5,
    base_size: float = 100.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate synthetic trade ticks from Hawkes event stream.

    Each event becomes a trade with a direction (buy/sell) and
    size modulated by inter-event spacing.

    Parameters
    ----------
    events : np.ndarray
        Hawkes event timestamps (trade times).
    base_mid : float
        Starting price.
    volatility : float
        Per-event price volatility.
    buy_prob : float
        Probability of a buy trade.
    base_size : float
        Base trade size.
    seed : int
        RNG seed.

    Returns
    -------
    pd.DataFrame
        Columns: time, price, size, side (1=buy, -1=sell), ofi_contribution.
    """
    rng = np.random.RandomState(seed)
    n = len(events)

    if n == 0:
        return pd.DataFrame(
            columns=["time", "price", "size", "side", "ofi_contribution"]
        )

    # Price random walk at trade times
    returns = rng.normal(0, volatility, n)
    prices = base_mid * np.exp(np.cumsum(returns))

    # Trade directions
    sides = np.where(rng.rand(n) < buy_prob, 1, -1)

    # Trade sizes (burstier periods → larger trades)
    inter_event = np.diff(events, prepend=0)
    # Shorter inter-event → larger cluster → larger size
    size_mult = np.clip(1.0 / (inter_event + 0.01), 0.5, 5.0)
    sizes = base_size * size_mult * (1 + rng.exponential(0.3, n))

    # OFI contribution: side * size
    ofi = sides * sizes

    return pd.DataFrame({
        "time": events,
        "price": prices,
        "size": sizes,
        "side": sides,
        "ofi_contribution": ofi,
    })
