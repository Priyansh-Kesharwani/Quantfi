"""
Unit tests for Deflated Sharpe Ratio (objective.py).

Verify DSR decreases as n_trials increases; sanity checks.
"""

import numpy as np
import pytest

from validation.objective import compute_dsr


class TestDSR:
    def test_dsr_decreases_with_more_trials(self):
        sharpes = [0.5, 0.6, 0.4]
        d1 = compute_dsr(sharpes, n_trials=10)
        d2 = compute_dsr(sharpes, n_trials=100)
        d3 = compute_dsr(sharpes, n_trials=1000)
        assert d1 >= d2
        assert d2 >= d3

    def test_empty_sharpes_returns_zero(self):
        assert compute_dsr([], n_trials=10) == 0.0
        assert compute_dsr([], n_trials=0) == 0.0

    def test_single_trial(self):
        d = compute_dsr([1.0], n_trials=1)
        assert d == 1.0

    def test_negative_observed_sharpe(self):
        d = compute_dsr([-0.5, -0.3], n_trials=50)
        assert d < 0
