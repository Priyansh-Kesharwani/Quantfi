"""
Phase A test fixtures — Synthetic series generators.

Provides deterministic FBM (fractional Brownian motion) and OU
(Ornstein-Uhlenbeck) series for unit-testing indicators.
"""

import numpy as np
import pandas as pd
from typing import Optional

def _cholesky_fbm(n: int, H: float) -> np.ndarray:
    """Generate FBM increments via Cholesky decomposition of the
    autocovariance matrix.  Deterministic given a fixed RNG state.
    """
    indices = np.arange(1, n + 1, dtype=np.float64)
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ti = indices[i]
            tj = indices[j]
            cov[i, j] = 0.5 * (
                ti ** (2 * H) + tj ** (2 * H) - abs(ti - tj) ** (2 * H)
            )

    cov += np.eye(n) * 1e-10

    L = np.linalg.cholesky(cov)
    z = np.random.randn(n)
    return L @ z

def fbm_series(
    n: int = 500,
    H: float = 0.7,
    seed: int = 42,
    start_price: float = 100.0,
    scale: float = 0.01,
) -> pd.DataFrame:
    """Generate a synthetic price DataFrame from FBM.

    Parameters
    ----------
    n : int
        Number of bars.
    H : float
        Hurst exponent (0 < H < 1).
    seed : int
        RNG seed for determinism.
    start_price : float
        Initial price level.
    scale : float
        Volatility scale factor for increments.

    Returns
    -------
    pd.DataFrame
        Columns: open, high, low, close, volume.
        Index: UTC DatetimeIndex.
    """
    np.random.seed(seed)
    increments = _cholesky_fbm(n, H) * scale
    log_prices = np.cumsum(increments)
    close = start_price * np.exp(log_prices)

    noise_hi = np.abs(np.random.randn(n)) * start_price * 0.005
    noise_lo = np.abs(np.random.randn(n)) * start_price * 0.005
    high = close + noise_hi
    low = close - noise_lo
    open_ = np.roll(close, 1)
    open_[0] = start_price
    volume = np.abs(np.random.randn(n) * 1e6 + 5e6)

    dates = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )

def ou_series(
    n: int = 500,
    theta: float = 0.15,
    mu: float = 100.0,
    sigma: float = 2.0,
    seed: int = 42,
    dt: float = 1.0,
) -> pd.DataFrame:
    """Generate a mean-reverting OU price series.

    dX = θ(μ − X) dt + σ dW

    Parameters
    ----------
    n : int
        Number of bars.
    theta : float
        Mean-reversion speed.
    mu : float
        Long-run mean.
    sigma : float
        Diffusion (volatility).
    seed : int
        RNG seed.
    dt : float
        Time step.

    Returns
    -------
    pd.DataFrame with columns open, high, low, close, volume.
    """
    np.random.seed(seed)
    x = np.zeros(n)
    x[0] = mu

    for t in range(1, n):
        dw = np.random.randn() * np.sqrt(dt)
        x[t] = x[t - 1] + theta * (mu - x[t - 1]) * dt + sigma * dw

    close = x
    noise_hi = np.abs(np.random.randn(n)) * sigma * 0.3
    noise_lo = np.abs(np.random.randn(n)) * sigma * 0.3
    high = close + noise_hi
    low = close - noise_lo
    open_ = np.roll(close, 1)
    open_[0] = mu
    volume = np.abs(np.random.randn(n) * 1e6 + 5e6)

    dates = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )

def poisson_events(
    rate: float = 1.0,
    T: float = 100.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate homogeneous Poisson process event times."""
    np.random.seed(seed)
    n_expected = int(rate * T * 1.5)
    inter_arrivals = np.random.exponential(1.0 / rate, size=n_expected)
    times = np.cumsum(inter_arrivals)
    return times[times <= T]

def hawkes_events(
    mu: float = 0.5,
    alpha: float = 0.3,
    beta: float = 1.0,
    T: float = 100.0,
    seed: int = 42,
) -> np.ndarray:
    """Simulate a univariate Hawkes process via Ogata's thinning method."""
    np.random.seed(seed)
    events = []
    t = 0.0
    lam_star = mu

    while t < T:
        u = np.random.rand()
        w = -np.log(u) / lam_star
        t = t + w
        if t >= T:
            break

        lam_t = mu
        for s in events:
            lam_t += alpha * np.exp(-beta * (t - s))

        d = np.random.rand()
        if d <= lam_t / lam_star:
            events.append(t)
            lam_star = lam_t + alpha
        else:
            lam_star = lam_t

    return np.array(events)
