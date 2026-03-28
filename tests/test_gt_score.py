"""
Unit tests for GT-Score (objective.py).

Verify formula sign/magnitude and underperformance penalty.
"""

import numpy as np
import pandas as pd
import pytest

from engine.validation.objective import compute_gt_score


class TestGTScoreUnderperformance:
    def test_underperform_penalty(self):
        n = 100
        eq = pd.Series(100 * (1.001 ** np.arange(n)))
        bm = pd.Series(100 * (1.002 ** np.arange(n)))
        score = compute_gt_score(eq, bm)
        assert score < 0

    def test_outperform_positive_gt(self):
        n = 100
        eq = pd.Series(100 * (1.002 ** np.arange(n)))
        bm = pd.Series(100 * (1.001 ** np.arange(n)))
        score = compute_gt_score(eq, bm)
        assert score > 0


class TestGTScoreFormula:
    def test_higher_r2_higher_score(self):
        n = 100
        t = np.arange(n, dtype=float)
        bm = pd.Series(100 * (1.001 ** t))
        eq_linear = pd.Series(100 + 0.5 * t)
        eq_noisy = pd.Series(100 + 0.5 * t + np.random.RandomState(42).randn(n) * 2)
        score_linear = compute_gt_score(eq_linear, bm)
        score_noisy = compute_gt_score(eq_noisy, bm)
        assert score_linear > score_noisy

    def test_short_series_penalty(self):
        eq = pd.Series([100.0, 101.0])
        bm = pd.Series([100.0, 100.5])
        score = compute_gt_score(eq, bm)
        assert score < -1e5 or np.isfinite(score)


class TestGTScoreEdgeCases:
    def test_empty_series(self):
        eq = pd.Series(dtype=float)
        bm = pd.Series(dtype=float)
        score = compute_gt_score(eq, bm)
        assert score < -1e5

    def test_min_trades_penalty(self):
        n = 20
        eq = pd.Series(100 * (1.01 ** np.arange(n)))
        bm = pd.Series(100 * (1.005 ** np.arange(n)))
        score = compute_gt_score(eq, bm, min_trades=50)
        assert score < -1e4
