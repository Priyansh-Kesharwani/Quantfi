"""Synthetic data generators for tests and offline scripts.

These were originally in backend/routes/crypto.py but the production backtest
route now always fetches real exchange data.  Tests and tuning scripts that
need deterministic / offline data should import from here instead.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from engine.crypto.calendar import bars_per_day

_TF_TO_PANDAS_FREQ = {
    "1m": "1min", "5m": "5min", "15m": "15min",
    "1h": "1h", "4h": "4h", "1d": "1D",
}

SYMBOL_PARAMS = {
    "BTC/USDT:USDT": {"base": 42000, "vol": 0.015, "drift": 0.0002},
    "ETH/USDT:USDT": {"base": 2500, "vol": 0.020, "drift": 0.0003},
    "SOL/USDT:USDT": {"base": 100, "vol": 0.030, "drift": 0.0004},
    "BNB/USDT:USDT": {"base": 320, "vol": 0.018, "drift": 0.0002},
}


def generate_synthetic_data(
    n: int,
    symbol: str,
    timeframe: str = "1h",
    seed: int = 42,
    start_date: Optional[str] = None,
) -> pd.DataFrame:
    """Generate multi-regime synthetic OHLCV with realistic trends and volatility."""
    if n <= 0:
        raise ValueError("n must be positive")

    params = SYMBOL_PARAMS.get(symbol, {"base": 1000, "vol": 0.015, "drift": 0.0002})
    base_price = params["base"]

    bpd = bars_per_day(timeframe)
    bpd_1h = bars_per_day("1h")
    tf_scale = math.sqrt(bpd_1h / bpd)

    base_vol = params["vol"] * tf_scale
    base_drift = params["drift"] * (bpd_1h / bpd)

    sym_hash = sum(ord(c) for c in symbol) % 1000
    rng = np.random.RandomState(seed + sym_hash)

    returns = np.zeros(n)
    i = 0
    while i < n:
        regime_len = max(40, int(rng.geometric(1 / 250)))
        end = min(i + regime_len, n)
        segment = end - i
        regime = rng.choice(
            ["bull", "bear", "range", "volatile"],
            p=[0.30, 0.20, 0.35, 0.15],
        )
        if regime == "bull":
            returns[i:end] = rng.normal(base_drift * 4, base_vol * 0.8, segment)
        elif regime == "bear":
            returns[i:end] = rng.normal(-base_drift * 3, base_vol, segment)
        elif regime == "range":
            returns[i:end] = rng.normal(0, base_vol * 0.5, segment)
        else:
            returns[i:end] = rng.normal(-base_drift, base_vol * 2.0, segment)
        i = end

    n_shocks = max(1, n // 400)
    shock_idx = rng.choice(n, size=n_shocks, replace=False)
    returns[shock_idx] += rng.normal(-0.04 * tf_scale, 0.015 * tf_scale, n_shocks)

    close = base_price * np.exp(np.cumsum(returns))

    intrabar = np.abs(returns) + rng.exponential(0.002 * tf_scale, n)
    high = close * (1 + intrabar * 0.6)
    low = close * (1 - intrabar * 0.6)
    open_ = close * (1 + rng.randn(n) * 0.002 * tf_scale)

    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    base_volume = 500 + rng.exponential(200, n)
    vol_spike = 1.0 + 3.0 * np.abs(returns) / 0.01
    volume = base_volume * vol_spike

    origin = start_date or "2023-01-01"
    freq = _TF_TO_PANDAS_FREQ.get(timeframe, "1h")
    idx = pd.date_range(origin, periods=n, freq=freq)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )

    assert not df.isna().any().any(), "Synthetic data contains NaN"
    assert (df["close"] > 0).all(), "Prices must be positive"
    assert (df["volume"] >= 0).all(), "Volume must be non-negative"
    assert df.index.is_monotonic_increasing, "Index must be monotonic"
    return df


def generate_synthetic_funding(
    n: int,
    symbol: str,
    timeframe: str = "1h",
    seed: int = 42,
    start_date: Optional[str] = None,
) -> pd.Series:
    sym_hash = sum(ord(c) for c in symbol) % 1000
    rng = np.random.RandomState(seed + sym_hash + 7)
    funding = rng.normal(0.0001, 0.0004, n)
    origin = start_date or "2023-01-01"
    freq = _TF_TO_PANDAS_FREQ.get(timeframe, "1h")
    return pd.Series(funding, index=pd.date_range(origin, periods=n, freq=freq))
