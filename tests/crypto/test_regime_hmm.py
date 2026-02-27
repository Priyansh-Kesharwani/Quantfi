"""Tests for HMM-based regime detection: covars_ shape, multi-init, no fallback, determinism."""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from crypto.regime.detector import (
    REGIME_RANGING,
    REGIME_STRESS,
    REGIME_TRENDING,
    CryptoRegimeConfig,
    CryptoRegimeDetector,
    _extract_state_variances,
    _fit_hmm_best_of_n,
)

warnings.filterwarnings("ignore")


def _make_ohlcv(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    returns = np.zeros(n)
    i = 0
    while i < n:
        seg = min(rng.randint(50, 300), n - i)
        regime = rng.choice(["bull", "bear", "range", "vol"], p=[0.3, 0.2, 0.35, 0.15])
        if regime == "bull":
            returns[i:i + seg] = rng.normal(0.001, 0.012, seg)
        elif regime == "bear":
            returns[i:i + seg] = rng.normal(-0.0008, 0.015, seg)
        elif regime == "range":
            returns[i:i + seg] = rng.normal(0, 0.006, seg)
        else:
            returns[i:i + seg] = rng.normal(-0.0003, 0.025, seg)
        i += seg
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(returns) * 0.6 + 0.002)
    low = close * (1 - np.abs(returns) * 0.6 - 0.002)
    high = np.maximum(high, close)
    low = np.minimum(low, close)
    idx = pd.date_range("2023-01-01", periods=n, freq="1h")
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": rng.exponential(500, n)}, index=idx)


class TestExtractStateVariances:
    """Verify _extract_state_variances handles both 2D and 3D covars_."""

    def test_map_states_handles_3d_covars(self):
        mock_model = MagicMock()
        covars_3d = np.zeros((3, 4, 4))
        for i in range(3):
            np.fill_diagonal(covars_3d[i], [0.1 * (i + 1)] * 4)
        mock_model.covars_ = covars_3d
        v = _extract_state_variances(mock_model)
        assert v.shape == (3,), f"Expected (3,), got {v.shape}"
        assert v[0] < v[1] < v[2]

    def test_map_states_handles_2d_covars(self):
        mock_model = MagicMock()
        covars_2d = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        mock_model.covars_ = covars_2d
        v = _extract_state_variances(mock_model)
        assert v.shape == (3,), f"Expected (3,), got {v.shape}"
        np.testing.assert_allclose(v, [0.3, 0.7, 1.1])


class TestHMMProducesAllRegimes:

    def test_hmm_produces_all_three_regimes(self):
        ohlcv = _make_ohlcv(2000)
        det = CryptoRegimeDetector(CryptoRegimeConfig(
            warmup_bars=500, rolling_window=500, refit_every=100,
            vol_window=96, cooldown_bars=3, circuit_breaker_dd=-0.25,
        ))
        regimes = det.fit_rolling(ohlcv)
        unique = set(regimes.unique())
        assert REGIME_TRENDING in unique, f"Missing TRENDING in {unique}"
        assert REGIME_RANGING in unique, f"Missing RANGING in {unique}"
        assert REGIME_STRESS in unique, f"Missing STRESS in {unique}"

    def test_hmm_no_heuristic_fallback(self):
        det = CryptoRegimeDetector()
        assert not hasattr(det, "_classify_heuristic") or not callable(
            getattr(det, "_classify_heuristic", None)
        ), "_classify_heuristic should be removed"


class TestMultipleInits:

    def test_hmm_multiple_inits_picks_best(self):
        rng = np.random.RandomState(0)
        X = np.column_stack([
            rng.randn(500),
            rng.randn(500) * 2,
            rng.randn(500) * 0.5,
            rng.exponential(1, 500),
        ])
        model = _fit_hmm_best_of_n(X, n_components=3, n_inits=10)
        assert model is not None
        assert model.n_components == 3
        score = model.score(X)
        assert np.isfinite(score)

    def test_hmm_refit_on_rolling_window(self):
        ohlcv = _make_ohlcv(2000)
        det = CryptoRegimeDetector(CryptoRegimeConfig(
            warmup_bars=400, rolling_window=400, refit_every=50,
            vol_window=48, cooldown_bars=3, circuit_breaker_dd=-0.30,
        ))
        regimes = det.fit_rolling(ohlcv)
        assert det._model is not None
        assert len(det._state_map) >= 2


class TestTwoStateFallback:

    def test_hmm_2state_fallback(self):
        rng = np.random.RandomState(99)
        close = 100 + np.cumsum(rng.normal(0, 0.001, 1000))
        high = close + 0.1
        low = close - 0.1
        idx = pd.date_range("2023-01-01", periods=1000, freq="1h")
        ohlcv = pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": np.ones(1000)}, index=idx)

        det = CryptoRegimeDetector(CryptoRegimeConfig(
            warmup_bars=300, rolling_window=300, refit_every=100,
            vol_window=48, cooldown_bars=3, circuit_breaker_dd=-0.50,
        ))
        regimes = det.fit_rolling(ohlcv)
        assert det._model is not None
        assert det._model.n_components in (2, 3)


class TestDeterminism:

    def test_regime_deterministic(self):
        ohlcv = _make_ohlcv(1500, seed=77)
        cfg = CryptoRegimeConfig(warmup_bars=400, rolling_window=400, refit_every=80, vol_window=48, cooldown_bars=3)
        r1 = CryptoRegimeDetector(cfg).fit_rolling(ohlcv)
        r2 = CryptoRegimeDetector(cfg).fit_rolling(ohlcv)
        pd.testing.assert_series_equal(r1, r2)


class TestFeaturesNoLookahead:

    def test_features_no_lookahead(self):
        ohlcv = _make_ohlcv(500, seed=10)
        det = CryptoRegimeDetector(CryptoRegimeConfig(warmup_bars=200, vol_window=48))
        features_full = det._prepare_features(ohlcv)
        features_partial = det._prepare_features(ohlcv.iloc[:300])
        np.testing.assert_allclose(
            features_full[:300], features_partial,
            atol=1e-10,
            err_msg="Features at bar t should only depend on data up to t"
        )

    def test_features_no_nan(self):
        ohlcv = _make_ohlcv(500)
        ohlcv.iloc[50:55, ohlcv.columns.get_loc("close")] = np.nan
        det = CryptoRegimeDetector()
        features = det._prepare_features(ohlcv)
        assert not np.any(np.isnan(features)), "Features should have no NaN"
