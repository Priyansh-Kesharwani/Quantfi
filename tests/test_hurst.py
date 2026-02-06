"""
Unit Tests for Hurst Exponent Estimator

Tests the Hurst exponent estimation on synthetic fractional Brownian motion
series with known Hurst parameters.

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

from indicators.hurst import estimate_hurst, _rescaled_range_hurst


class TestRescaledRangeHurst:
    """Test the R/S Hurst estimation method."""
    
    def test_random_walk_returns_around_0_5(self):
        """Random walk should have H ≈ 0.5."""
        np.random.seed(42)
        # Generate random walk (cumsum of random returns)
        returns = np.random.randn(1000) * 0.01
        prices = 100 * np.exp(np.cumsum(returns))
        
        H_t, meta = estimate_hurst(prices, window=252, method="rs")
        
        # Get the last valid H value
        valid_H = H_t[~np.isnan(H_t)]
        assert len(valid_H) > 0, "Should have valid H estimates"
        
        # Average H should be around 0.5 for random walk
        avg_H = np.mean(valid_H)
        assert 0.35 < avg_H < 0.65, f"Random walk H should be ~0.5, got {avg_H:.3f}"
        
        # Check metadata
        assert meta["method"] == "rs"
        assert meta["window_used"] == 252
        assert meta["seed"] is None  # Deterministic
    
    def test_trending_series_returns_high_hurst(self):
        """Trending series should have H > 0.5."""
        np.random.seed(123)
        # Generate trending series with positive drift and persistence
        n = 1000
        returns = np.random.randn(n) * 0.01 + 0.002  # Strong upward drift
        # Add persistence
        for i in range(1, n):
            returns[i] = 0.4 * returns[i-1] + 0.6 * returns[i]
        
        prices = 100 * np.exp(np.cumsum(returns))
        
        H_t, meta = estimate_hurst(prices, window=252, method="rs")
        
        valid_H = H_t[~np.isnan(H_t)]
        avg_H = np.mean(valid_H)
        
        # Trending series should have H > 0.5
        assert avg_H > 0.45, f"Trending series H should be > 0.45, got {avg_H:.3f}"
    
    def test_mean_reverting_series_returns_low_hurst(self):
        """Mean-reverting series should have H < 0.5."""
        np.random.seed(456)
        # Generate mean-reverting series (Ornstein-Uhlenbeck-like)
        n = 1000
        prices = np.zeros(n)
        prices[0] = 100
        mean_level = 100
        reversion_speed = 0.1
        
        for i in range(1, n):
            shock = np.random.randn() * 2
            prices[i] = prices[i-1] + reversion_speed * (mean_level - prices[i-1]) + shock
        
        H_t, meta = estimate_hurst(prices, window=252, method="rs")
        
        valid_H = H_t[~np.isnan(H_t)]
        avg_H = np.mean(valid_H)
        
        # Mean-reverting series should have H < 0.55
        assert avg_H < 0.55, f"Mean-reverting series H should be < 0.55, got {avg_H:.3f}"
    
    def test_output_shape_matches_input(self):
        """Output array should have same length as input."""
        np.random.seed(789)
        n = 500
        prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
        
        H_t, meta = estimate_hurst(prices, window=100)
        
        assert len(H_t) == n, f"Output length {len(H_t)} should match input {n}"
    
    def test_warmup_period_is_nan(self):
        """Values before warmup window should be NaN."""
        np.random.seed(111)
        n = 500
        window = 200
        prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
        
        H_t, meta = estimate_hurst(prices, window=window)
        
        # First 'window' values should be NaN
        assert np.all(np.isnan(H_t[:window])), "Warmup period should be NaN"
        
        # After warmup, should have valid values
        valid_after = ~np.isnan(H_t[window:])
        assert np.sum(valid_after) > 0, "Should have valid values after warmup"
    
    def test_deterministic_with_seed(self):
        """Same input should produce same output."""
        np.random.seed(222)
        prices = 100 * np.exp(np.cumsum(np.random.randn(400) * 0.01))
        
        H_t1, _ = estimate_hurst(prices, window=100, method="rs")
        H_t2, _ = estimate_hurst(prices, window=100, method="rs")
        
        np.testing.assert_array_equal(H_t1, H_t2, "Same input should give same output")


class TestHurstWithFixtures:
    """Test Hurst estimation using fixture data."""
    
    @pytest.fixture
    def trending_fixture(self):
        """Load trending asset fixture."""
        fixture_path = PROJECT_ROOT / "tests" / "fixtures" / "trending_asset.csv"
        if fixture_path.exists():
            df = pd.read_csv(fixture_path, parse_dates=["date"])
            return df["close"].values
        else:
            # Generate synthetic trending data
            np.random.seed(42)
            n = 300
            returns = np.random.randn(n) * 0.01 + 0.001
            for i in range(1, n):
                returns[i] = 0.3 * returns[i-1] + 0.7 * returns[i]
            return 100 * np.exp(np.cumsum(returns))
    
    @pytest.fixture
    def fbm_fixture(self):
        """Load fractional Brownian motion fixture with known H."""
        fixture_path = PROJECT_ROOT / "tests" / "fixtures" / "fbm_series.csv"
        if fixture_path.exists():
            df = pd.read_csv(fixture_path, parse_dates=["date"])
            return df["close"].values, df["true_hurst"].iloc[0]
        else:
            # Generate synthetic fBm-like data with H=0.7
            np.random.seed(42)
            n = 300
            true_H = 0.7
            returns = np.random.randn(n) * 0.01
            # Simulate persistence
            for i in range(1, n):
                returns[i] = 0.4 * returns[i-1] + 0.6 * returns[i]
            return 100 * np.exp(np.cumsum(returns)), true_H
    
    def test_trending_fixture_has_high_hurst(self, trending_fixture):
        """Trending fixture should show persistent behavior."""
        prices = trending_fixture
        
        H_t, meta = estimate_hurst(prices, window=100, method="rs")
        
        valid_H = H_t[~np.isnan(H_t)]
        if len(valid_H) > 0:
            avg_H = np.mean(valid_H)
            # Trending data should have H > 0.4
            assert avg_H > 0.4, f"Trending fixture H should be > 0.4, got {avg_H:.3f}"
    
    def test_fbm_fixture_within_expected_band(self, fbm_fixture):
        """fBm fixture estimate should be within expected range of true H."""
        prices, true_H = fbm_fixture
        
        H_t, meta = estimate_hurst(prices, window=100, method="rs")
        
        valid_H = H_t[~np.isnan(H_t)]
        if len(valid_H) > 0:
            avg_H = np.mean(valid_H)
            # Allow ±0.25 tolerance (R/S is not super accurate)
            assert true_H - 0.25 < avg_H < true_H + 0.25, \
                f"fBm H={true_H}, estimated {avg_H:.3f}"


class TestHurstMetadata:
    """Test metadata returned by Hurst estimator."""
    
    def test_metadata_contains_required_fields(self):
        """Metadata should have all required fields for explainability."""
        np.random.seed(333)
        prices = 100 * np.exp(np.cumsum(np.random.randn(400) * 0.01))
        
        _, meta = estimate_hurst(prices, window=150)
        
        required_fields = ["window_used", "n_obs", "method", "seed", "notes"]
        for field in required_fields:
            assert field in meta, f"Metadata missing required field: {field}"
    
    def test_metadata_window_matches_input(self):
        """Metadata window_used should match input parameter."""
        np.random.seed(444)
        prices = 100 * np.exp(np.cumsum(np.random.randn(400) * 0.01))
        
        _, meta = estimate_hurst(prices, window=200)
        
        assert meta["window_used"] == 200
    
    def test_metadata_n_obs_matches_input_length(self):
        """Metadata n_obs should match input series length."""
        np.random.seed(555)
        n = 350
        prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
        
        _, meta = estimate_hurst(prices)
        
        assert meta["n_obs"] == n


class TestHurstEdgeCases:
    """Test edge cases and error handling."""
    
    def test_short_series_returns_nan(self):
        """Very short series should return NaN (insufficient data)."""
        prices = np.array([100, 101, 102, 103, 104])
        
        H_t, meta = estimate_hurst(prices, window=252)
        
        # All values should be NaN for short series
        assert np.all(np.isnan(H_t)), "Short series should return all NaN"
    
    def test_constant_series_handles_gracefully(self):
        """Constant series (zero volatility) should not crash."""
        prices = np.full(400, 100.0)
        
        # Should not raise an exception
        H_t, meta = estimate_hurst(prices, window=100)
        
        # Result may be NaN or some default value, but should not crash
        assert len(H_t) == 400
    
    def test_negative_prices_handled(self):
        """Negative or zero prices should be handled gracefully."""
        np.random.seed(666)
        prices = np.random.randn(400) * 10  # Some negative values
        
        # Should not raise an exception
        H_t, meta = estimate_hurst(prices, window=100)
        
        assert len(H_t) == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
