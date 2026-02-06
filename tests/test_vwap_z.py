"""
Unit Tests for VWAP Z-Score Indicator

Tests the VWAP-based valuation Z-score calculation.

NON-NEGOTIABLE:
- All tests use seeded RNG for reproducibility
- No network access
- Deterministic fixtures

Author: Phase 1 Implementation
Date: 2026-02-07
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from indicators.vwap_z import compute_vwap_z, vwap_undervaluation_score


class TestVWAPCalculation:
    """Test VWAP Z-score calculation."""
    
    def test_price_below_vwap_returns_negative_z(self):
        """Price below VWAP should return negative Z-score."""
        np.random.seed(42)
        n = 100
        
        # Create prices that end below VWAP
        prices = np.linspace(110, 90, n)  # Declining prices
        volumes = np.full(n, 1_000_000)
        
        Z_t, meta = compute_vwap_z(prices, volumes, window=20)
        
        # Last Z should be negative (price below fair value)
        valid_Z = Z_t[~np.isnan(Z_t)]
        assert len(valid_Z) > 0
        
        # Recent Z values should be negative for declining price
        recent_Z = valid_Z[-10:]
        assert np.mean(recent_Z) < 0, f"Declining prices should have negative Z, got {np.mean(recent_Z):.3f}"
    
    def test_price_above_vwap_returns_positive_z(self):
        """Price above VWAP should return positive Z-score."""
        np.random.seed(43)
        n = 100
        
        # Create prices that end above VWAP
        prices = np.linspace(90, 110, n)  # Rising prices
        volumes = np.full(n, 1_000_000)
        
        Z_t, meta = compute_vwap_z(prices, volumes, window=20)
        
        valid_Z = Z_t[~np.isnan(Z_t)]
        assert len(valid_Z) > 0
        
        # Recent Z values should be positive for rising price
        recent_Z = valid_Z[-10:]
        assert np.mean(recent_Z) > 0, f"Rising prices should have positive Z, got {np.mean(recent_Z):.3f}"
    
    def test_stable_price_returns_near_zero_z(self):
        """Stable prices around VWAP should have Z near zero."""
        np.random.seed(44)
        n = 100
        
        # Stable prices with small noise
        base_price = 100
        prices = base_price + np.random.randn(n) * 0.5
        volumes = np.full(n, 1_000_000) + np.random.randn(n) * 100_000
        volumes = np.maximum(volumes, 500_000)
        
        Z_t, meta = compute_vwap_z(prices, volumes, window=20)
        
        valid_Z = Z_t[~np.isnan(Z_t)]
        assert len(valid_Z) > 0
        
        # Z should be relatively small for stable prices
        assert np.abs(np.mean(valid_Z)) < 2, f"Stable prices should have small Z, got {np.mean(valid_Z):.3f}"
    
    def test_output_shape_matches_input(self):
        """Output should have same length as input."""
        n = 200
        prices = np.linspace(100, 110, n)
        volumes = np.full(n, 1_000_000)
        
        Z_t, meta = compute_vwap_z(prices, volumes, window=20)
        
        assert len(Z_t) == n


class TestVWAPFallback:
    """Test SMA fallback when volume is missing."""
    
    def test_no_volume_uses_sma_fallback(self):
        """Missing volume should trigger SMA fallback."""
        np.random.seed(45)
        prices = np.linspace(100, 110, 100)
        
        # No volume provided
        Z_t, meta = compute_vwap_z(prices, volume_series=None, window=20)
        
        assert meta["method"] == "sma_fallback"
        assert "TODO" in meta["notes"] or "volume" in meta["notes"].lower()
    
    def test_insufficient_volume_uses_sma_fallback(self):
        """Insufficient volume data should trigger SMA fallback."""
        prices = np.linspace(100, 110, 100)
        volumes = np.full(100, np.nan)  # All NaN volume
        
        Z_t, meta = compute_vwap_z(prices, volumes, window=20)
        
        assert meta["method"] == "sma_fallback"
    
    def test_sma_fallback_produces_valid_output(self):
        """SMA fallback should still produce valid Z-scores."""
        np.random.seed(46)
        n = 100
        prices = np.linspace(90, 110, n)  # Rising
        
        Z_t, meta = compute_vwap_z(prices, volume_series=None, window=20)
        
        valid_Z = Z_t[~np.isnan(Z_t)]
        assert len(valid_Z) > 0
        
        # Rising prices should have positive Z even with SMA fallback
        recent_Z = valid_Z[-10:]
        assert np.mean(recent_Z) > 0


class TestUndervaluationScore:
    """Test the undervaluation score wrapper."""
    
    def test_undervaluation_inverts_z_score(self):
        """Undervaluation score should invert Z polarity."""
        np.random.seed(47)
        n = 100
        
        # Declining prices (negative Z = undervalued)
        prices = np.linspace(110, 90, n)
        volumes = np.full(n, 1_000_000)
        
        # Get raw Z
        Z_t, _ = compute_vwap_z(prices, volumes, window=20)
        
        # Get undervaluation score (inverted Z)
        U_t, meta = vwap_undervaluation_score(prices, volumes, window=20)
        
        valid_Z = Z_t[~np.isnan(Z_t)]
        valid_U = U_t[~np.isnan(U_t)]
        
        # U should be approximately -Z
        if len(valid_Z) > 0 and len(valid_U) > 0:
            np.testing.assert_array_almost_equal(valid_U, -valid_Z, decimal=5)
    
    def test_undervalued_asset_has_positive_score(self):
        """Undervalued asset (price < VWAP) should have positive undervaluation score."""
        np.random.seed(48)
        n = 100
        
        # Declining prices = undervalued
        prices = np.linspace(110, 90, n)
        volumes = np.full(n, 1_000_000)
        
        U_t, meta = vwap_undervaluation_score(prices, volumes, window=20)
        
        valid_U = U_t[~np.isnan(U_t)]
        recent_U = valid_U[-10:]
        
        # Undervalued should have positive score
        assert np.mean(recent_U) > 0, f"Undervalued asset should have positive U, got {np.mean(recent_U):.3f}"


class TestVWAPWithFixtures:
    """Test VWAP using fixture data."""
    
    @pytest.fixture
    def price_volume_fixture(self):
        """Load fixture with price and volume."""
        fixture_path = PROJECT_ROOT / "tests" / "fixtures" / "synthetic_prices.csv"
        if fixture_path.exists():
            df = pd.read_csv(fixture_path, parse_dates=["date"])
            return df["close"].values, df["volume"].values
        else:
            # Generate synthetic data
            np.random.seed(42)
            n = 50
            prices = 100 + np.cumsum(np.random.randn(n) * 2)
            volumes = np.full(n, 1_000_000) + np.random.randn(n) * 200_000
            return prices, np.maximum(volumes, 500_000)
    
    def test_fixture_produces_valid_z_scores(self, price_volume_fixture):
        """Fixture data should produce valid Z-scores."""
        prices, volumes = price_volume_fixture
        
        Z_t, meta = compute_vwap_z(prices, volumes, window=10)
        
        valid_Z = Z_t[~np.isnan(Z_t)]
        
        # Should have some valid values
        assert len(valid_Z) > 0
        
        # Z-scores should be reasonable (within ±10 for typical data)
        assert np.all(np.abs(valid_Z) < 10), "Z-scores should be reasonable"


class TestVWAPMetadata:
    """Test metadata returned by VWAP functions."""
    
    def test_metadata_contains_required_fields(self):
        """Metadata should have all required fields."""
        prices = np.linspace(100, 110, 100)
        volumes = np.full(100, 1_000_000)
        
        _, meta = compute_vwap_z(prices, volumes, window=20)
        
        required_fields = ["window_used", "n_obs", "method", "seed", "notes"]
        for field in required_fields:
            assert field in meta, f"Missing field: {field}"
    
    def test_vwap_method_when_volume_provided(self):
        """Method should be 'vwap' when valid volume is provided."""
        prices = np.linspace(100, 110, 100)
        volumes = np.full(100, 1_000_000)
        
        _, meta = compute_vwap_z(prices, volumes, window=20)
        
        assert meta["method"] == "vwap"


class TestVWAPEdgeCases:
    """Test edge cases."""
    
    def test_short_series_handles_gracefully(self):
        """Short series should not crash."""
        prices = np.array([100, 101, 102])
        volumes = np.array([1_000_000, 1_100_000, 1_050_000])
        
        Z_t, meta = compute_vwap_z(prices, volumes, window=20)
        
        assert len(Z_t) == 3
        # Should be all NaN for short series
        assert np.all(np.isnan(Z_t))
    
    def test_zero_volume_handled(self):
        """Zero volume should not crash."""
        prices = np.linspace(100, 110, 50)
        volumes = np.zeros(50)  # All zero
        
        Z_t, meta = compute_vwap_z(prices, volumes, window=10)
        
        # Should fall back to SMA
        assert meta["method"] == "sma_fallback"
    
    def test_window_larger_than_data_returns_nan(self):
        """Window larger than data should return all NaN."""
        prices = np.linspace(100, 110, 30)
        volumes = np.full(30, 1_000_000)
        
        Z_t, meta = compute_vwap_z(prices, volumes, window=50)
        
        # All NaN because window > data length
        assert np.all(np.isnan(Z_t))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
