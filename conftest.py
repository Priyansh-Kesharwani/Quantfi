"""Root conftest: shared fixtures for all test suites."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_ohlcv():
    """500-bar synthetic OHLCV DataFrame with realistic structure."""
    rng = np.random.RandomState(42)
    n = 500
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, n)))
    high = close * (1 + rng.uniform(0.001, 0.02, n))
    low = close * (1 - rng.uniform(0.001, 0.02, n))
    opn = low + (high - low) * rng.uniform(0.2, 0.8, n)
    volume = rng.randint(100_000, 5_000_000, n).astype(float)
    dates = pd.bdate_range("2020-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


@pytest.fixture
def synthetic_close(synthetic_ohlcv):
    """Close price array from synthetic OHLCV."""
    return synthetic_ohlcv["Close"].values
