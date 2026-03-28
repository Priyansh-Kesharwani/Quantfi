"""Tests for crypto.scoring.directional_scorer."""

import numpy as np
import pandas as pd
import pytest

from engine.crypto.scoring.directional_scorer import (
    CryptoDirectionalScorer,
    ScoringConfig,
    _self_calibrating_tanh,
    verify_score_reachability,
)


def _make_ohlcv(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="1h")
    close = 50_000.0 + np.cumsum(rng.randn(n) * 100)
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


class TestSelfCalibratingTanh:
    def test_output_bounded(self):
        signal = pd.Series(np.random.randn(200))
        result = _self_calibrating_tanh(signal, 50)
        assert result.max() <= 100.0
        assert result.min() >= -100.0

    def test_one_sigma_maps_to_76(self):
        signal = pd.Series(np.random.randn(300))
        result = _self_calibrating_tanh(signal, 100)
        mask = signal.abs().between(0.8, 1.2)
        if mask.sum() > 5:
            avg_abs = result[mask].abs().mean()
            assert 50 < avg_abs < 90


class TestCryptoDirectionalScorer:
    @pytest.fixture
    def scorer(self):
        return CryptoDirectionalScorer(ScoringConfig(compression_window=50))

    def test_output_range(self, scorer):
        ohlcv = _make_ohlcv(500)
        scores = scorer.compute(ohlcv)
        assert len(scores) == 500
        assert scores.max() <= 200
        assert scores.min() >= -200

    def test_no_crash_on_short_data(self, scorer):
        ohlcv = _make_ohlcv(30)
        scores = scorer.compute(ohlcv)
        assert len(scores) == 30

    def test_with_funding_rates(self, scorer):
        ohlcv = _make_ohlcv(500)
        funding = pd.Series(
            np.random.RandomState(42).normal(0.0001, 0.0005, 500),
            index=ohlcv.index,
        )
        scores = scorer.compute(ohlcv, funding_rates=funding)
        assert len(scores) == 500

    def test_with_open_interest(self, scorer):
        ohlcv = _make_ohlcv(500)
        oi = pd.Series(
            np.cumsum(np.random.RandomState(42).randn(500) * 1000) + 100_000,
            index=ohlcv.index,
        )
        scores = scorer.compute(ohlcv, open_interest=oi)
        assert len(scores) == 500

    def test_uniform_weights_mode(self, scorer):
        ohlcv = _make_ohlcv(500)
        scores = scorer.compute_with_uniform_weights(ohlcv)
        assert len(scores) == 500

    def test_different_configs_different_scores(self):
        ohlcv = _make_ohlcv(500)
        s1 = CryptoDirectionalScorer(ScoringConfig(rsi_oversold=20, compression_window=50))
        s2 = CryptoDirectionalScorer(ScoringConfig(rsi_oversold=35, compression_window=50))
        scores1 = s1.compute_with_uniform_weights(ohlcv)
        scores2 = s2.compute_with_uniform_weights(ohlcv)
        assert not np.allclose(scores1.values, scores2.values)


class TestScoreReachability:
    def test_reachable_threshold(self):
        scores = pd.Series(np.random.RandomState(42).randn(1000) * 30)
        result = verify_score_reachability(scores, 40)
        assert "entry_pct" in result
        assert result["entry_pct"] > 0

    def test_unreachable_threshold(self):
        scores = pd.Series(np.random.RandomState(42).randn(1000) * 5)
        result = verify_score_reachability(scores, 90)
        assert result["entry_pct"] < 0.5
        assert not result["ok"]

    def test_too_loose_threshold(self):
        scores = pd.Series(np.random.RandomState(42).randn(1000) * 50)
        result = verify_score_reachability(scores, 5)
        assert result["entry_pct"] > 30
        assert not result["ok"]

    def test_empty_scores(self):
        result = verify_score_reachability(pd.Series(dtype=float), 40)
        assert not result["ok"]
