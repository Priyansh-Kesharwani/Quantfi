"""Tests for crypto.regime.detector."""

import numpy as np
import pandas as pd
import pytest

from crypto.regime.detector import (
    REGIME_RANGING,
    REGIME_STRESS,
    REGIME_TRENDING,
    CryptoRegimeConfig,
    CryptoRegimeDetector,
    default_regime_config,
)


def _make_ohlcv(n: int = 1000, trend: bool = False, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="1h")
    close = 50_000.0 + np.cumsum(rng.randn(n) * 100)
    if trend:
        close += np.linspace(0, 5000, n)
    return pd.DataFrame(
        {
            "open": close + rng.randn(n) * 20,
            "high": close + abs(rng.randn(n) * 80),
            "low": close - abs(rng.randn(n) * 80),
            "close": close,
            "volume": 1000 + rng.rand(n) * 500,
        },
        index=idx,
    )


class TestDefaultRegimeConfig:
    def test_1h_config(self):
        cfg = default_regime_config("1h")
        assert cfg.n_states == 3
        assert cfg.vol_window > 0
        assert cfg.cooldown_bars >= 3

    def test_different_timeframes(self):
        cfg_1h = default_regime_config("1h")
        cfg_1d = default_regime_config("1d")
        assert cfg_1h.vol_window > cfg_1d.vol_window


class TestCryptoRegimeDetector:
    @pytest.fixture
    def small_config(self):
        return CryptoRegimeConfig(
            warmup_bars=100,
            rolling_window=200,
            refit_every=50,
            vol_window=30,
            cooldown_bars=3,
        )

    def test_basic_regime_detection(self, small_config):
        ohlcv = _make_ohlcv(300)
        det = CryptoRegimeDetector(small_config)
        labels = det.fit_rolling(ohlcv)
        assert len(labels) == 300
        unique = set(labels.unique())
        assert unique.issubset({REGIME_TRENDING, REGIME_RANGING, REGIME_STRESS})

    def test_short_data_returns_ranging(self, small_config):
        ohlcv = _make_ohlcv(50)
        det = CryptoRegimeDetector(small_config)
        labels = det.fit_rolling(ohlcv)
        assert (labels == REGIME_RANGING).all()

    def test_circuit_breaker_triggers(self, small_config):
        ohlcv = _make_ohlcv(300)
        ohlcv.iloc[250:, ohlcv.columns.get_loc("close")] = ohlcv["close"].iloc[200] * 0.80
        det = CryptoRegimeDetector(small_config)
        labels = det.fit_rolling(ohlcv)
        assert REGIME_STRESS in labels.values

    def test_hysteresis_prevents_flicker(self, small_config):
        small_config.cooldown_bars = 10
        ohlcv = _make_ohlcv(300)
        det = CryptoRegimeDetector(small_config)
        labels = det.fit_rolling(ohlcv)
        transitions = (labels != labels.shift(1)).sum()
        assert transitions < len(labels) * 0.3

    def test_no_lookahead(self, small_config):
        ohlcv = _make_ohlcv(300)
        det = CryptoRegimeDetector(small_config)
        labels_full = det.fit_rolling(ohlcv)
        det2 = CryptoRegimeDetector(small_config)
        labels_partial = det2.fit_rolling(ohlcv.iloc[:200])
        overlap = labels_full.iloc[:200].values == labels_partial.values
        assert overlap.mean() > 0.8

    def test_regime_distribution_reasonable(self):
        """No single regime should dominate more than 70% of bars."""
        from tests.crypto.synthetic import generate_synthetic_data

        ohlcv = generate_synthetic_data(2000, "BTC/USDT:USDT")
        det = CryptoRegimeDetector(CryptoRegimeConfig(
            warmup_bars=500, rolling_window=500, refit_every=100,
            vol_window=96, cooldown_bars=3, circuit_breaker_dd=-0.25,
        ))
        labels = det.fit_rolling(ohlcv)
        vc = labels.value_counts(normalize=True)
        for regime, pct in vc.items():
            assert pct < 0.70, f"{regime} dominates at {pct:.0%}"

    def test_circuit_breaker_rolling_window(self):
        """Circuit breaker uses rolling (not expanding) peak."""
        ohlcv = _make_ohlcv(500, seed=10)
        det = CryptoRegimeDetector(CryptoRegimeConfig(
            warmup_bars=100, rolling_window=200, refit_every=50,
            vol_window=30, cooldown_bars=3, circuit_breaker_dd=-0.10,
        ))
        prices = ohlcv["close"]
        labels_init = pd.Series(REGIME_RANGING, index=ohlcv.index)
        result = det._apply_circuit_breaker(prices, labels_init)
        stress_count = (result == REGIME_STRESS).sum()
        assert stress_count < len(result), "Not everything should be STRESS"

    def test_large_dataset_regime_stability(self):
        """On 5000 bars, regimes don't degenerate to a single label."""
        from tests.crypto.synthetic import generate_synthetic_data

        ohlcv = generate_synthetic_data(5000, "ETH/USDT:USDT")
        det = CryptoRegimeDetector(CryptoRegimeConfig(
            warmup_bars=1000, rolling_window=1000, refit_every=200,
            vol_window=168, cooldown_bars=5, circuit_breaker_dd=-0.25,
        ))
        labels = det.fit_rolling(ohlcv)
        unique = set(labels.unique())
        assert len(unique) >= 2, f"Expected at least 2 regimes, got {unique}"
