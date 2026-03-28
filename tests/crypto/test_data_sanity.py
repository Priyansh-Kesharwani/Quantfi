"""Data sanity tests: OHLCV invariants, NaN handling, length alignment."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from tests.crypto.synthetic import (
    SYMBOL_PARAMS,
    generate_synthetic_data as _generate_synthetic_data,
    generate_synthetic_funding as _generate_synthetic_funding,
)
from engine.crypto.regime.detector import CryptoRegimeConfig, CryptoRegimeDetector
from engine.crypto.scoring.directional_scorer import CryptoDirectionalScorer, ScoringConfig
from engine.crypto.services.backtest_service import CryptoBacktestConfig, CryptoBacktestService

SYMBOLS = list(SYMBOL_PARAMS.keys())


class TestSyntheticDataInvariants:

    @pytest.mark.parametrize("symbol", SYMBOLS)
    def test_synthetic_data_ohlcv_invariants(self, symbol):
        df = _generate_synthetic_data(2000, symbol)
        assert (df["high"] >= df["close"]).all(), "high must be >= close"
        assert (df["high"] >= df["open"]).all(), "high must be >= open"
        assert (df["low"] <= df["close"]).all(), "low must be <= close"
        assert (df["low"] <= df["open"]).all(), "low must be <= open"
        assert (df["high"] >= df["low"]).all(), "high must be >= low"

    @pytest.mark.parametrize("symbol", SYMBOLS)
    def test_synthetic_data_no_nan(self, symbol):
        df = _generate_synthetic_data(2000, symbol)
        assert not df.isna().any().any(), f"NaN found for {symbol}"

    def test_synthetic_data_monotonic_index(self):
        df = _generate_synthetic_data(1000, "BTC/USDT:USDT")
        assert df.index.is_monotonic_increasing

    def test_synthetic_data_positive_prices(self):
        for sym in SYMBOLS:
            df = _generate_synthetic_data(2000, sym)
            for col in ["open", "high", "low", "close"]:
                assert (df[col] > 0).all(), f"{col} must be positive for {sym}"

    def test_synthetic_funding_bounded(self):
        for sym in SYMBOLS:
            f = _generate_synthetic_funding(2000, sym)
            assert (f.abs() < 0.01).all(), f"Funding rate too large for {sym}"
            assert not f.isna().any(), f"NaN in funding for {sym}"


class TestBacktestInputValidation:

    def test_backtest_input_validation_nan_ohlcv(self):
        """Backtest handles NaN in OHLCV gracefully (fills, no crash)."""
        ohlcv = _generate_synthetic_data(500, "BTC/USDT:USDT")
        funding = _generate_synthetic_funding(500, "BTC/USDT:USDT")
        ohlcv.iloc[100:105, ohlcv.columns.get_loc("close")] = np.nan
        ohlcv["close"] = ohlcv["close"].ffill().bfill()

        cfg = CryptoBacktestConfig(
            compression_window=60, ic_window=60, ic_horizon=10, funding_window=80,
            regime_config=CryptoRegimeConfig(warmup_bars=100, rolling_window=100, refit_every=50, vol_window=48),
        )
        svc = CryptoBacktestService()
        result = svc.run(ohlcv, cfg, funding_rates=funding)
        assert np.isfinite(result["sharpe"])

    def test_scores_no_nan_output(self):
        for sym in SYMBOLS:
            ohlcv = _generate_synthetic_data(1000, sym)
            funding = _generate_synthetic_funding(1000, sym)
            scorer = CryptoDirectionalScorer(ScoringConfig(compression_window=60, funding_window=80))
            scores = scorer.compute_with_uniform_weights(ohlcv, funding, None)
            assert not scores.isna().any(), f"NaN in scores for {sym}"

    def test_regime_no_nan_output(self):
        for sym in SYMBOLS:
            ohlcv = _generate_synthetic_data(1000, sym)
            det = CryptoRegimeDetector(CryptoRegimeConfig(
                warmup_bars=200, rolling_window=200, refit_every=50, vol_window=48,
            ))
            regimes = det.fit_rolling(ohlcv)
            assert not regimes.isna().any(), f"NaN regime labels for {sym}"
            for r in regimes.unique():
                assert r in ("TRENDING", "RANGING", "STRESS"), f"Unknown regime: {r}"
