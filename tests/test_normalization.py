"""
Unit Tests for Normalization Pipeline

Tests the expanding ECDF → Z → Sigmoid normalization pipeline.

NON-NEGOTIABLE:
- All tests use seeded RNG for reproducibility
- No network access
- Deterministic fixtures

Author: Phase 1 Implementation
Date: 2026-02-07
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from indicators.normalization import (
    expanding_percentile,
    percentile_to_z,
    z_to_sigmoid,
    polarity_align,
    normalize_to_score,
    batch_normalize
)


class TestExpandingPercentile:
    """Test expanding ECDF percentile calculation."""
    
    def test_no_lookahead(self):
        """Percentiles should only use past data (no lookahead)."""
        np.random.seed(42)
        values = np.random.randn(500)
        
        pct_t, meta = expanding_percentile(values, min_obs=50)
        
        # Check a specific point
        test_idx = 200
        current_value = values[test_idx]
        historical = values[:test_idx]
        
        # Manual percentile calculation
        expected_pct = np.sum(historical <= current_value) / len(historical)
        expected_pct = np.clip(expected_pct, 0.001, 0.999)
        
        # Should match
        np.testing.assert_almost_equal(pct_t[test_idx], expected_pct, decimal=3)
    
    def test_adding_datapoint_updates_percentiles_deterministically(self):
        """Adding one datapoint should deterministically update percentiles."""
        np.random.seed(43)
        
        # Initial data
        initial = np.random.randn(200)
        pct_initial, _ = expanding_percentile(initial, min_obs=50)
        
        # Add one more datapoint
        extended = np.append(initial, 0.5)  # Add a specific value
        pct_extended, _ = expanding_percentile(extended, min_obs=50)
        
        # Historical percentiles should be the same
        np.testing.assert_array_equal(
            pct_initial[:200],
            pct_extended[:200],
            "Historical percentiles should not change"
        )
        
        # New point should have a valid percentile
        assert not np.isnan(pct_extended[200])
    
    def test_warmup_period_is_nan(self):
        """First min_obs values should be NaN."""
        np.random.seed(44)
        values = np.random.randn(300)
        min_obs = 100
        
        pct_t, meta = expanding_percentile(values, min_obs=min_obs)
        
        # First min_obs should be NaN
        assert np.all(np.isnan(pct_t[:min_obs]))
        
        # After min_obs, should have valid values
        assert not np.all(np.isnan(pct_t[min_obs:]))
    
    def test_output_in_valid_range(self):
        """Percentiles should be in (0, 1), not exact 0 or 1."""
        np.random.seed(45)
        values = np.random.randn(500)
        
        pct_t, meta = expanding_percentile(values, min_obs=50)
        
        valid_pct = pct_t[~np.isnan(pct_t)]
        
        # Should be > 0 and < 1 (clipped)
        assert np.all(valid_pct > 0), "Percentiles should be > 0"
        assert np.all(valid_pct < 1), "Percentiles should be < 1"
    
    def test_extreme_value_gets_high_percentile(self):
        """An extreme high value should have percentile near 1."""
        np.random.seed(46)
        # Normal values followed by an extreme
        values = np.random.randn(200)
        values = np.append(values, 100)  # Extreme outlier
        
        pct_t, _ = expanding_percentile(values, min_obs=50)
        
        # Last value (extreme) should have high percentile
        assert pct_t[-1] > 0.99


class TestPercentileToZ:
    """Test percentile to Z-score conversion."""
    
    def test_median_percentile_returns_zero_z(self):
        """Percentile of 0.5 should map to Z ≈ 0."""
        pct = np.array([0.5])
        z = percentile_to_z(pct)
        
        np.testing.assert_almost_equal(z[0], 0, decimal=5)
    
    def test_high_percentile_returns_positive_z(self):
        """High percentile should map to positive Z."""
        pct = np.array([0.9])
        z = percentile_to_z(pct)
        
        assert z[0] > 0
    
    def test_low_percentile_returns_negative_z(self):
        """Low percentile should map to negative Z."""
        pct = np.array([0.1])
        z = percentile_to_z(pct)
        
        assert z[0] < 0
    
    def test_handles_edge_values(self):
        """Should handle edge values without producing inf."""
        pct = np.array([0.001, 0.999])
        z = percentile_to_z(pct)
        
        assert np.all(np.isfinite(z))
    
    def test_preserves_nan(self):
        """NaN percentiles should remain NaN."""
        pct = np.array([0.5, np.nan, 0.7])
        z = percentile_to_z(pct)
        
        assert np.isnan(z[1])


class TestZToSigmoid:
    """Test Z-score to sigmoid transformation."""
    
    def test_zero_z_returns_half(self):
        """Z = 0 should map to sigmoid = 0.5."""
        z = np.array([0.0])
        sig = z_to_sigmoid(z)
        
        np.testing.assert_almost_equal(sig[0], 0.5, decimal=5)
    
    def test_positive_z_returns_greater_than_half(self):
        """Positive Z should map to sigmoid > 0.5."""
        z = np.array([1.0, 2.0, 3.0])
        sig = z_to_sigmoid(z)
        
        assert np.all(sig > 0.5)
    
    def test_negative_z_returns_less_than_half(self):
        """Negative Z should map to sigmoid < 0.5."""
        z = np.array([-1.0, -2.0, -3.0])
        sig = z_to_sigmoid(z)
        
        assert np.all(sig < 0.5)
    
    def test_output_bounded_zero_one(self):
        """Sigmoid output should be in (0, 1)."""
        z = np.array([-100, -10, -1, 0, 1, 10, 100])
        sig = z_to_sigmoid(z)
        
        assert np.all(sig > 0)
        assert np.all(sig < 1)
    
    def test_steepness_parameter_k(self):
        """Higher k should make transition steeper."""
        z = np.array([1.0])
        
        sig_k1 = z_to_sigmoid(z, k=1.0)
        sig_k2 = z_to_sigmoid(z, k=2.0)
        
        # k=2 should give value closer to 1 for positive Z
        assert sig_k2[0] > sig_k1[0]
    
    def test_preserves_nan(self):
        """NaN Z should remain NaN."""
        z = np.array([1.0, np.nan, -1.0])
        sig = z_to_sigmoid(z)
        
        assert np.isnan(sig[1])


class TestPolarityAlign:
    """Test polarity alignment function."""
    
    def test_higher_is_favorable_keeps_original(self):
        """When higher is favorable, score unchanged."""
        score = np.array([0.2, 0.5, 0.8])
        aligned = polarity_align(score, higher_is_favorable=True)
        
        np.testing.assert_array_equal(score, aligned)
    
    def test_higher_is_not_favorable_inverts(self):
        """When higher is not favorable, score is inverted."""
        score = np.array([0.2, 0.5, 0.8])
        aligned = polarity_align(score, higher_is_favorable=False)
        
        expected = np.array([0.8, 0.5, 0.2])
        np.testing.assert_array_equal(aligned, expected)


class TestNormalizeToScore:
    """Test full normalization pipeline."""
    
    def test_output_in_zero_one_range(self):
        """Final score should be in [0, 1]."""
        np.random.seed(47)
        raw_values = np.random.randn(500)
        
        score, meta = normalize_to_score(raw_values, min_obs=50)
        
        valid_scores = score[~np.isnan(score)]
        
        assert np.all(valid_scores >= 0), "Scores should be >= 0"
        assert np.all(valid_scores <= 1), "Scores should be <= 1"
    
    def test_median_value_gets_neutral_score(self):
        """Median-ish value should get score near 0.5."""
        np.random.seed(48)
        # Values centered around 0
        raw_values = np.random.randn(500)
        
        score, meta = normalize_to_score(raw_values, min_obs=50)
        
        valid_scores = score[~np.isnan(score)]
        
        # Average score should be near 0.5 for symmetric distribution
        avg_score = np.mean(valid_scores)
        assert 0.4 < avg_score < 0.6, f"Average score should be ~0.5, got {avg_score:.3f}"
    
    def test_high_raw_value_gets_high_score_when_favorable(self):
        """High raw value should get high score when higher_is_favorable=True."""
        np.random.seed(49)
        # Mostly normal, but end with high values
        raw_values = np.random.randn(400)
        raw_values = np.append(raw_values, np.array([5.0] * 10))  # High values at end
        
        score, meta = normalize_to_score(raw_values, min_obs=50, higher_is_favorable=True)
        
        # Last scores should be high
        assert score[-1] > 0.8
    
    def test_deterministic_with_same_input(self):
        """Same input should produce same output."""
        np.random.seed(50)
        raw_values = np.random.randn(300)
        
        score1, _ = normalize_to_score(raw_values, min_obs=50)
        score2, _ = normalize_to_score(raw_values, min_obs=50)
        
        np.testing.assert_array_equal(score1, score2)
    
    def test_metadata_contains_config(self):
        """Metadata should contain configuration used."""
        raw_values = np.random.randn(300)
        
        _, meta = normalize_to_score(raw_values, min_obs=75, k=1.5, higher_is_favorable=False)
        
        assert meta["min_obs"] == 75
        assert meta["sigmoid_k"] == 1.5
        assert meta["higher_is_favorable"] == False


class TestBatchNormalize:
    """Test batch normalization of multiple indicators."""
    
    def test_normalizes_all_indicators(self):
        """Should normalize all provided indicators."""
        np.random.seed(51)
        indicators = {
            "hurst": np.random.uniform(0.3, 0.7, 300),
            "vwap_z": np.random.randn(300),
            "volatility": np.random.uniform(0.1, 0.4, 300)
        }
        
        normalized, metas = batch_normalize(indicators)
        
        assert "hurst" in normalized
        assert "vwap_z" in normalized
        assert "volatility" in normalized
        
        # All should be in [0, 1]
        for name, values in normalized.items():
            valid = values[~np.isnan(values)]
            assert np.all(valid >= 0) and np.all(valid <= 1)
    
    def test_respects_per_indicator_config(self):
        """Should use per-indicator configuration."""
        np.random.seed(52)
        indicators = {
            "positive_good": np.random.randn(300),
            "negative_good": np.random.randn(300)
        }
        
        config = {
            "positive_good": {"higher_is_favorable": True},
            "negative_good": {"higher_is_favorable": False}
        }
        
        normalized, metas = batch_normalize(indicators, config)
        
        assert metas["positive_good"]["higher_is_favorable"] == True
        assert metas["negative_good"]["higher_is_favorable"] == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
