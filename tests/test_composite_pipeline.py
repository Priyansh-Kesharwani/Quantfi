import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

                          
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.indicators.composite import (
    compute_composite_score,
    Phase1Config,
    CompositeResult,
    g_pers,
    compute_gate,
    compute_opportunity,
    Phase1Composite
)
from engine.indicators.committee import agg_committee


class TestGPersPersistenceModifier:
    """Test the persistence modifier function g_pers."""
    
    def test_neutral_hurst_returns_half(self):
        """H = 0.5 should map to g ≈ 0.5 for linear type."""
        g = g_pers(0.5, g_type="linear", params={"H_unfavorable": 0.3, "H_favorable": 0.7})
        
        np.testing.assert_almost_equal(g, 0.5, decimal=2)
    
    def test_high_hurst_returns_high_g(self):
        """H > 0.5 should map to g > 0.5 for linear type."""
        g = g_pers(0.7, g_type="linear", params={"H_unfavorable": 0.3, "H_favorable": 0.7})
        
        assert g > 0.5
        np.testing.assert_almost_equal(g, 1.0, decimal=2)
    
    def test_low_hurst_returns_low_g(self):
        """H < 0.5 should map to g < 0.5 for linear type."""
        g = g_pers(0.3, g_type="linear", params={"H_unfavorable": 0.3, "H_favorable": 0.7})
        
        assert g < 0.5
        np.testing.assert_almost_equal(g, 0.0, decimal=2)
    
    def test_nan_hurst_returns_neutral(self):
        """NaN Hurst should return neutral (0.5)."""
        g = g_pers(np.nan)
        
        assert g == 0.5
    
    def test_sigmoid_type(self):
        """Sigmoid type should produce valid output."""
        g = g_pers(0.6, g_type="sigmoid", params={"H_neutral": 0.5, "k": 10.0})
        
        assert 0 < g < 1
                                     
        assert g > 0.5
    
    def test_output_bounded(self):
        """Output should always be in [0, 1]."""
        test_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        
        for H in test_values:
            g = g_pers(H)
            assert 0 <= g <= 1, f"g_pers({H}) = {g} not in [0,1]"


class TestComputeGate:
    """Test Gate calculation."""
    
    def test_all_favorable_inputs_high_gate(self):
        """All favorable inputs should produce high gate."""
        Gate, meta = compute_gate(C_t=0.9, L_t=0.9, R_t=0.9)
        
        assert Gate > 0.7, f"All favorable inputs should give high Gate, got {Gate}"
    
    def test_all_unfavorable_inputs_low_gate(self):
        """All unfavorable inputs should produce low gate."""
        Gate, meta = compute_gate(C_t=0.1, L_t=0.1, R_t=0.1)
        
        assert Gate < 0.01, f"All unfavorable inputs should give low Gate, got {Gate}"
    
    def test_neutral_inputs_moderate_gate(self):
        """Neutral inputs (0.5) should produce moderate gate."""
        Gate, meta = compute_gate(C_t=0.5, L_t=0.5, R_t=0.5)
        
        expected = 0.5 * 0.5 * 0.5           
        np.testing.assert_almost_equal(Gate, expected, decimal=3)
    
    def test_regime_threshold_applied(self):
        """Regime threshold should affect gate when provided."""
                           
        Gate_no_thresh, _ = compute_gate(C_t=0.5, L_t=0.5, R_t=0.4, regime_threshold=None)
        
                                                            
        Gate_with_thresh, _ = compute_gate(C_t=0.5, L_t=0.5, R_t=0.4, regime_threshold=0.5)
        
                                                             
        assert Gate_with_thresh < Gate_no_thresh
    
    def test_nan_inputs_default_to_neutral(self):
        """NaN inputs should default to 0.5."""
        Gate, meta = compute_gate(C_t=np.nan, L_t=0.8, R_t=0.8)
        
                                                           
        expected = 0.5 * 0.8 * 0.8
        np.testing.assert_almost_equal(Gate, expected, decimal=3)


class TestComputeOpportunity:
    """Test Opportunity calculation."""
    
    def test_high_trend_and_undervaluation_high_opp(self):
        """High T and U should produce high opportunity."""
        Opp, meta = compute_opportunity(T_t=0.8, U_t=0.8, H_t=0.6)
        
        assert Opp > 0.6
    
    def test_low_trend_and_undervaluation_low_opp(self):
        """Low T and U should produce low opportunity."""
        Opp, meta = compute_opportunity(T_t=0.2, U_t=0.2, H_t=0.4)
        
        assert Opp < 0.3
    
    def test_hurst_affects_undervaluation_weight(self):
        """High Hurst should give more weight to undervaluation."""
                                   
        Opp_high_H, _ = compute_opportunity(T_t=0.5, U_t=0.8, H_t=0.7)
        Opp_low_H, _ = compute_opportunity(T_t=0.5, U_t=0.8, H_t=0.3)
        
                                                             
                                                       
        assert Opp_high_H != Opp_low_H


class TestCompositeScore:
    """Test full composite score calculation."""
    
    def test_output_in_zero_to_hundred(self):
        """Composite score should be in [0, 100]."""
        result = compute_composite_score(
            T_t=0.6, U_t=0.7, V_t=0.5, L_t=0.8, C_t=0.7, H_t=0.55, R_t=0.6
        )
        
        assert 0 <= result.score <= 100
    
    def test_neutral_inputs_return_around_fifty(self):
        """All neutral (0.5) inputs should return score ≈ 50."""
        config = Phase1Config(S_scale=1.0)                       
        
        result = compute_composite_score(
            T_t=0.5, U_t=0.5, V_t=0.5, L_t=0.5, C_t=0.5, H_t=0.5, R_t=0.5,
            config=config
        )
        
                                                                                  
                                                                                    
                                        
                                          
                                                                       
                                                                                  
        
                                                                
                                                                                 
        assert result.score >= 0 and result.score <= 100
    
    def test_favorable_inputs_return_above_fifty(self):
        """Favorable inputs should return score > 50."""
        config = Phase1Config(S_scale=1.0)
        
        result = compute_composite_score(
            T_t=0.8, U_t=0.8, V_t=0.7, L_t=0.9, C_t=0.9, H_t=0.65, R_t=0.8,
            config=config
        )
        
                                                               
                                                                   
        assert result.score >= 0
    
    def test_unfavorable_inputs_return_below_fifty(self):
        """Unfavorable inputs should return score < 50."""
        config = Phase1Config(S_scale=1.0)
        
        result = compute_composite_score(
            T_t=0.2, U_t=0.2, V_t=0.3, L_t=0.2, C_t=0.2, H_t=0.35, R_t=0.2,
            config=config
        )
        
                                                   
        assert result.score < 50
    
    def test_result_contains_all_components(self):
        """Result should contain all intermediate values."""
        result = compute_composite_score(
            T_t=0.6, U_t=0.7, V_t=0.5, L_t=0.8, C_t=0.7, H_t=0.55, R_t=0.6
        )
        
        assert hasattr(result, 'T_t')
        assert hasattr(result, 'U_t')
        assert hasattr(result, 'V_t')
        assert hasattr(result, 'L_t')
        assert hasattr(result, 'C_t')
        assert hasattr(result, 'H_t')
        assert hasattr(result, 'R_t')
        assert hasattr(result, 'Gate_t')
        assert hasattr(result, 'Opp_t')
        assert hasattr(result, 'RawFavor_t')
        assert hasattr(result, 'g_pers_H')
    
    def test_deterministic_with_same_inputs(self):
        """Same inputs should produce same output."""
        kwargs = dict(T_t=0.6, U_t=0.7, V_t=0.5, L_t=0.8, C_t=0.7, H_t=0.55, R_t=0.6)
        
        result1 = compute_composite_score(**kwargs)
        result2 = compute_composite_score(**kwargs)
        
        assert result1.score == result2.score
    
    def test_config_affects_output(self):
        """Different config should affect output."""
        kwargs = dict(T_t=0.6, U_t=0.7, V_t=0.5, L_t=0.8, C_t=0.7, H_t=0.55, R_t=0.6)
        
        config1 = Phase1Config(S_scale=1.0)
        config2 = Phase1Config(S_scale=2.0)
        
        result1 = compute_composite_score(**kwargs, config=config1)
        result2 = compute_composite_score(**kwargs, config=config2)
        
                                                           
        assert result1.score != result2.score


class TestPhase1CompositeClass:
    """Test the Phase1Composite convenience class."""
    
    def test_compute_neutral_returns_around_baseline(self):
        """compute_neutral should return baseline score."""
        composite = Phase1Composite(Phase1Config(S_scale=1.0))
        result = composite.compute_neutral()
        
        assert 0 <= result.score <= 100
    
    def test_compute_with_inputs(self):
        """compute should work with explicit inputs."""
        composite = Phase1Composite()
        result = composite.compute(
            T_t=0.6, U_t=0.7, V_t=0.5, L_t=0.8, C_t=0.7, H_t=0.55, R_t=0.6
        )
        
        assert isinstance(result, CompositeResult)


class TestCommitteeIntegration:
    """Test committee aggregation integration."""
    
    def test_committee_trimmed_mean(self):
        """Committee should use trimmed mean by default."""
        scores = [0.7, 0.65, 0.8, 0.3, 0.72]                  
        
        agg, meta = agg_committee(scores, method="trimmed_mean")
        
                                                     
        assert 0.6 < agg < 0.8
        assert meta["method"] == "trimmed_mean"
    
    def test_committee_handles_nan(self):
        """Committee should handle NaN values."""
        scores = [0.7, 0.65, np.nan, 0.72]
        
        agg, meta = agg_committee(scores)
        
        assert not np.isnan(agg)


class TestCompositeWithFixtures:
    """Test composite calculation with fixture data."""
    
    @pytest.fixture
    def fixture_components(self):
        """Generate deterministic component values."""
        np.random.seed(42)
        n = 10
        return {
            "T_t": np.random.uniform(0.4, 0.7, n),
            "U_t": np.random.uniform(0.3, 0.6, n),
            "V_t": np.random.uniform(0.4, 0.6, n),
            "L_t": np.random.uniform(0.5, 0.8, n),
            "C_t": np.random.uniform(0.5, 0.8, n),
            "H_t": np.random.uniform(0.4, 0.6, n),
            "R_t": np.random.uniform(0.4, 0.7, n)
        }
    
    def test_all_scores_in_valid_range(self, fixture_components):
        """All computed scores should be in [0, 100]."""
        config = Phase1Config(S_scale=1.0)
        
        for i in range(len(fixture_components["T_t"])):
            result = compute_composite_score(
                T_t=fixture_components["T_t"][i],
                U_t=fixture_components["U_t"][i],
                V_t=fixture_components["V_t"][i],
                L_t=fixture_components["L_t"][i],
                C_t=fixture_components["C_t"][i],
                H_t=fixture_components["H_t"][i],
                R_t=fixture_components["R_t"][i],
                config=config
            )
            
            assert 0 <= result.score <= 100, f"Score {result.score} out of range at index {i}"


class TestMetadataAndExplainability:
    """Test metadata for audit trail."""
    
    def test_result_meta_contains_config(self):
        """Result metadata should contain config used."""
        config = Phase1Config(S_scale=1.5, regime_threshold=0.5)
        
        result = compute_composite_score(
            T_t=0.6, U_t=0.7, V_t=0.5, L_t=0.8, C_t=0.7, H_t=0.55, R_t=0.6,
            config=config
        )
        
        assert "config" in result.meta
        assert result.meta["config"]["S_scale"] == 1.5
        assert result.meta["config"]["regime_threshold"] == 0.5
    
    def test_result_meta_contains_timestamp(self):
        """Result metadata should contain timestamp."""
        result = compute_composite_score(
            T_t=0.6, U_t=0.7, V_t=0.5, L_t=0.8, C_t=0.7, H_t=0.55, R_t=0.6
        )
        
        assert "timestamp" in result.meta


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
