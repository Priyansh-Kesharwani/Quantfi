"""Phase A — Lorentzian Distance Classifier unit tests."""

import pytest
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from indicators.ldc import LDC, lorentzian_distance, build_templates_from_labels

class TestLorentzianDistance:

    def test_identical_vectors_zero_distance(self):
        x = np.array([1.0, 2.0, 3.0])
        assert lorentzian_distance(x, x) == pytest.approx(0.0)

    def test_non_negative(self):
        np.random.seed(42)
        for _ in range(20):
            x = np.random.randn(5)
            y = np.random.randn(5)
            assert lorentzian_distance(x, y) >= 0

    def test_symmetric(self):
        x = np.array([1.0, 3.0, -2.0])
        y = np.array([4.0, 0.0, 1.0])
        d_xy = lorentzian_distance(x, y)
        d_yx = lorentzian_distance(y, x)
        assert d_xy == pytest.approx(d_yx)

    def test_gamma_scales_distance(self):
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        d_small_gamma = lorentzian_distance(x, y, gamma=np.array([0.1, 0.1]))
        d_large_gamma = lorentzian_distance(x, y, gamma=np.array([10.0, 10.0]))
        assert d_small_gamma > d_large_gamma

    def test_formula_correctness(self):
        """Verify the formula: Σ log(1 + ((x_i − y_i)/γ_i)²)."""
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 5.0])
        gamma = np.array([1.0, 2.0])
        expected = np.log(1 + ((1 - 3) / 1) ** 2) + np.log(1 + ((2 - 5) / 2) ** 2)
        assert lorentzian_distance(x, y, gamma) == pytest.approx(expected)

class TestLDC:

    @pytest.fixture
    def simple_ldc(self):
        ldc = LDC(kappa=1.0)
        ldc.fit({
            "bull": np.array([1.0, 1.0, 1.0]),
            "bear": np.array([-1.0, -1.0, -1.0]),
        })
        return ldc

    def test_fit_requires_bull_bear(self):
        ldc = LDC()
        with pytest.raises(ValueError):
            ldc.fit({"up": np.array([1.0])})

    def test_score_before_fit_raises(self):
        ldc = LDC()
        with pytest.raises(RuntimeError):
            ldc.score(np.array([0.0]))

    def test_bull_vector_scores_high(self, simple_ldc):
        """A point equal to the bull template should score > 0.5."""
        score = simple_ldc.score(np.array([1.0, 1.0, 1.0]))
        assert score > 0.5

    def test_bear_vector_scores_low(self, simple_ldc):
        """A point equal to the bear template should score < 0.5."""
        score = simple_ldc.score(np.array([-1.0, -1.0, -1.0]))
        assert score < 0.5

    def test_midpoint_scores_near_half(self, simple_ldc):
        """Equidistant point should score ≈ 0.5."""
        score = simple_ldc.score(np.array([0.0, 0.0, 0.0]))
        assert 0.4 < score < 0.6

    def test_output_bounded_zero_one(self, simple_ldc):
        np.random.seed(42)
        for _ in range(50):
            x = np.random.randn(3) * 5
            s = simple_ldc.score(x)
            assert 0 < s < 1

    def test_kappa_affects_sharpness(self):
        templates = {
            "bull": np.array([1.0, 1.0]),
            "bear": np.array([-1.0, -1.0]),
        }
        ldc_gentle = LDC(kappa=0.1)
        ldc_gentle.fit(templates)
        ldc_sharp = LDC(kappa=10.0)
        ldc_sharp.fit(templates)

        x = np.array([0.5, 0.5])
        s_gentle = ldc_gentle.score(x)
        s_sharp = ldc_sharp.score(x)
        assert s_sharp > s_gentle

    def test_score_batch(self, simple_ldc):
        X = np.array([
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0],
            [0.0, 0.0, 0.0],
        ])
        scores = simple_ldc.score_batch(X)
        assert scores.shape == (3,)
        assert scores[0] > 0.5
        assert scores[1] < 0.5

    def test_deterministic(self, simple_ldc):
        x = np.array([0.3, -0.2, 0.8])
        s1 = simple_ldc.score(x)
        s2 = simple_ldc.score(x)
        assert s1 == s2

class TestLDCMotifSeparation:

    def test_roc_auc_above_threshold(self):
        """LDC should separate bull/bear motifs with AUC > 0.8."""
        np.random.seed(42)
        n_per_class = 100
        d = 5

        bull_features = np.random.randn(n_per_class, d) + 1.0
        bear_features = np.random.randn(n_per_class, d) - 1.0

        features = np.vstack([bull_features, bear_features])
        labels = np.array([1] * n_per_class + [0] * n_per_class)

        templates = build_templates_from_labels(features, labels)
        ldc = LDC(kappa=1.0)
        ldc.fit(templates)

        scores = ldc.score_batch(features)

        from tests._auc_helper import manual_roc_auc
        auc = manual_roc_auc(labels, scores)
        assert auc > 0.8, f"AUC = {auc:.3f}, expected > 0.8"

class TestLDCSerialisation:

    def test_round_trip(self):
        ldc = LDC(kappa=2.0)
        ldc.fit({
            "bull": np.array([1.0, 2.0]),
            "bear": np.array([-1.0, -2.0]),
        })
        d = ldc.to_dict()
        ldc2 = LDC.from_dict(d)
        x = np.array([0.5, 0.5])
        assert ldc.score(x) == pytest.approx(ldc2.score(x))

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
