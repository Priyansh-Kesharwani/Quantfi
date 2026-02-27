"""Tests for crypto.adapters (symbol resolver, data quality, funding alignment)."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from crypto.adapters.symbol_resolver import parse_symbol, resolve_symbol
from crypto.adapters.data_quality import (
    DataQualityReport,
    DataQualityValidator,
    align_funding_to_bars,
)


class TestSymbolResolver:
    def test_perpetual_btc(self):
        assert resolve_symbol("BTC") == "BTC/USDT:USDT"

    def test_perpetual_eth_busd(self):
        assert resolve_symbol("ETH", "BUSD") == "ETH/BUSD:BUSD"

    def test_spot(self):
        assert resolve_symbol("BTC", "USDT", "spot") == "BTC/USDT"

    def test_case_insensitive(self):
        assert resolve_symbol("btc", "usdt") == "BTC/USDT:USDT"

    def test_parse_perpetual(self):
        result = parse_symbol("BTC/USDT:USDT")
        assert result["base"] == "BTC"
        assert result["quote"] == "USDT"
        assert result["instrument"] == "perpetual"

    def test_parse_spot(self):
        result = parse_symbol("ETH/USDT")
        assert result["base"] == "ETH"
        assert result["instrument"] == "spot"


class TestDataQualityValidator:
    @pytest.fixture
    def validator(self):
        return DataQualityValidator()

    @pytest.fixture
    def clean_1h_data(self):
        idx = pd.date_range("2024-01-01", periods=100, freq="1h")
        rng = np.random.RandomState(42)
        return pd.DataFrame(
            {
                "open": 50_000 + rng.randn(100) * 100,
                "high": 50_200 + rng.randn(100) * 100,
                "low": 49_800 + rng.randn(100) * 100,
                "close": 50_000 + rng.randn(100) * 100,
                "volume": 1000 + rng.rand(100) * 500,
            },
            index=idx,
        )

    def test_clean_data_passes(self, validator, clean_1h_data):
        report = validator.validate(clean_1h_data, "1h")
        assert report.is_acceptable
        assert report.missing_bars == 0
        assert report.max_consecutive_gap == 0

    def test_empty_dataframe_fails(self, validator):
        df = pd.DataFrame()
        report = validator.validate(df, "1h")
        assert not report.is_acceptable

    def test_detects_gaps(self, validator):
        idx = pd.date_range("2024-01-01", periods=10, freq="1h")
        idx = idx.delete([3, 4, 5])
        rng = np.random.RandomState(42)
        df = pd.DataFrame(
            {
                "open": rng.randn(7) + 50_000,
                "high": rng.randn(7) + 50_200,
                "low": rng.randn(7) + 49_800,
                "close": rng.randn(7) + 50_000,
                "volume": rng.rand(7) * 1000,
            },
            index=idx,
        )
        report = validator.validate(df, "1h")
        assert report.missing_bars > 0
        assert report.max_consecutive_gap >= 2

    def test_detects_nan_values(self, validator, clean_1h_data):
        clean_1h_data.iloc[5, 0] = np.nan
        report = validator.validate(clean_1h_data, "1h")
        assert any("NaN" in w for w in report.warnings)

    def test_mark_untradeable(self, clean_1h_data):
        clean_1h_data.iloc[3, 0] = np.nan
        tradeable = DataQualityValidator.mark_untradeable(clean_1h_data)
        assert not tradeable.iloc[3]
        assert tradeable.iloc[0]
        assert tradeable.sum() == 99


class TestFundingAlignment:
    def test_alignment_1h_bars(self):
        bar_index = pd.date_range("2024-01-01 00:00", periods=24, freq="1h")
        funding_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2024-01-01 00:00", "2024-01-01 08:00", "2024-01-01 16:00"]
                ),
                "fundingRate": [0.0001, 0.0002, -0.0001],
            }
        )
        aligned = align_funding_to_bars(funding_df, bar_index, "1h")
        assert len(aligned) == 24
        assert aligned.iloc[0] == pytest.approx(0.0001)
        zero_bars = aligned[aligned == 0.0]
        assert len(zero_bars) < 24

    def test_empty_funding(self):
        bar_index = pd.date_range("2024-01-01", periods=10, freq="1h")
        funding_df = pd.DataFrame(columns=["timestamp", "fundingRate"])
        aligned = align_funding_to_bars(funding_df, bar_index, "1h")
        assert (aligned == 0.0).all()
        assert len(aligned) == 10
