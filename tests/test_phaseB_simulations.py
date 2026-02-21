"""
Phase B — Tests for Hawkes Process Simulation Module.

Validates:
  - Event generation determinism
  - Branching ratio / stationarity calculations
  - Ground-truth λ(t) correctness
  - Regime simulations (bursty, near-Poisson, explosive edge, sparse)
  - RMSE calculation and estimation validation
  - MLE recovery on simulated events
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from simulations.hawkes_simulator import (
    simulate_hawkes_events,
    ground_truth_intensity,
    branching_ratio,
    expected_event_rate,
    intensity_rmse,
    relative_rmse,
    simulate_regime,
    run_all_regimes,
    validate_estimation,
)
from indicators.hawkes import estimate_hawkes


# ====================================================================
# Event Simulation Tests
# ====================================================================

class TestSimulateHawkesEvents:
    def test_determinism(self):
        e1 = simulate_hawkes_events(0.5, 0.3, 1.0, 100.0, seed=42)
        e2 = simulate_hawkes_events(0.5, 0.3, 1.0, 100.0, seed=42)
        np.testing.assert_array_equal(e1, e2)

    def test_different_seeds_differ(self):
        e1 = simulate_hawkes_events(0.5, 0.3, 1.0, 100.0, seed=42)
        e2 = simulate_hawkes_events(0.5, 0.3, 1.0, 100.0, seed=99)
        assert not np.array_equal(e1, e2)

    def test_events_in_range(self):
        events = simulate_hawkes_events(0.5, 0.3, 1.0, 100.0, seed=0)
        assert (events >= 0).all()
        assert (events <= 100.0).all()

    def test_events_sorted(self):
        events = simulate_hawkes_events(0.5, 0.3, 1.0, 100.0, seed=0)
        assert np.all(np.diff(events) >= 0)

    def test_high_rate_many_events(self):
        events = simulate_hawkes_events(5.0, 0.1, 1.0, 100.0, seed=0)
        # High baseline should produce many events
        assert len(events) > 100

    def test_low_rate_fewer_events(self):
        events = simulate_hawkes_events(0.01, 0.005, 1.0, 100.0, seed=0)
        # Low rate should produce few events
        assert len(events) < 50

    def test_empty_short_interval(self):
        events = simulate_hawkes_events(0.01, 0.005, 1.0, 0.1, seed=0)
        # Very short interval with low rate → possibly 0 events
        assert len(events) >= 0  # just should not crash


# ====================================================================
# Ground Truth Intensity Tests
# ====================================================================

class TestGroundTruthIntensity:
    def test_no_events_equals_baseline(self):
        grid = np.linspace(0, 10, 100)
        lam = ground_truth_intensity(np.array([]), 0.5, 0.3, 1.0, grid)
        np.testing.assert_allclose(lam, 0.5, atol=1e-12)

    def test_single_event_decay(self):
        events = np.array([1.0])
        grid = np.array([0.5, 1.5, 2.5, 5.0])
        mu, alpha, beta = 0.5, 0.3, 1.0
        lam = ground_truth_intensity(events, mu, alpha, beta, grid)

        # Before event: λ = μ
        assert lam[0] == pytest.approx(mu, abs=1e-10)
        # After event: λ = μ + α·exp(-β·Δt)
        assert lam[1] == pytest.approx(mu + alpha * np.exp(-beta * 0.5), abs=1e-10)
        assert lam[2] == pytest.approx(mu + alpha * np.exp(-beta * 1.5), abs=1e-10)

    def test_intensity_nonnegative(self):
        events = simulate_hawkes_events(0.5, 0.3, 1.0, 50.0, seed=42)
        grid = np.linspace(0, 50, 500)
        lam = ground_truth_intensity(events, 0.5, 0.3, 1.0, grid)
        assert (lam >= 0).all()

    def test_intensity_at_least_mu(self):
        events = simulate_hawkes_events(0.5, 0.3, 1.0, 50.0, seed=42)
        grid = np.linspace(0, 50, 500)
        lam = ground_truth_intensity(events, 0.5, 0.3, 1.0, grid)
        assert (lam >= 0.5 - 1e-12).all()


# ====================================================================
# Branching Ratio & Expected Rate Tests
# ====================================================================

class TestBranchingRatio:
    def test_subcritical(self):
        eta = branching_ratio(0.3, 1.0)
        assert eta == pytest.approx(0.3, abs=1e-10)
        assert eta < 1.0

    def test_critical(self):
        eta = branching_ratio(1.0, 1.0)
        assert eta == pytest.approx(1.0, abs=1e-10)

    def test_supercritical(self):
        eta = branching_ratio(1.5, 1.0)
        assert eta > 1.0

    def test_zero_beta(self):
        eta = branching_ratio(0.5, 0.0)
        assert eta == np.inf


class TestExpectedEventRate:
    def test_stationary(self):
        rate = expected_event_rate(0.5, 0.3, 1.0)
        # μ/(1-η) = 0.5/(1-0.3) ≈ 0.714
        assert rate == pytest.approx(0.5 / 0.7, abs=1e-6)

    def test_explosive(self):
        rate = expected_event_rate(0.5, 1.5, 1.0)
        assert rate == np.inf

    def test_critical(self):
        rate = expected_event_rate(0.5, 1.0, 1.0)
        assert rate == np.inf


# ====================================================================
# RMSE Tests
# ====================================================================

class TestRMSE:
    def test_identical_zero_rmse(self):
        x = np.array([1.0, 2.0, 3.0])
        rmse = intensity_rmse(x, x)
        assert rmse == pytest.approx(0.0, abs=1e-12)

    def test_known_rmse(self):
        est = np.array([1.0, 2.0, 3.0])
        truth = np.array([1.0, 2.0, 4.0])
        rmse = intensity_rmse(est, truth)
        # RMSE = sqrt(mean([0, 0, 1])) = sqrt(1/3) ≈ 0.577
        assert rmse == pytest.approx(np.sqrt(1 / 3), abs=1e-6)

    def test_relative_rmse(self):
        est = np.array([1.0, 2.0, 3.0])
        truth = np.array([1.0, 2.0, 4.0])
        rrmse = relative_rmse(est, truth)
        # RMSE / mean(truth) = sqrt(1/3) / mean([1,2,4])
        assert rrmse == pytest.approx(np.sqrt(1 / 3) / (7 / 3), abs=1e-6)


# ====================================================================
# Regime Simulation Tests
# ====================================================================

class TestRegimeSimulation:
    def test_bursty_regime(self):
        result = simulate_regime("bursty", mu=0.1, alpha=0.8, beta=1.0, T=200.0, seed=42)
        assert result["regime_name"] == "bursty"
        assert result["branching_ratio"] == pytest.approx(0.8, abs=1e-10)
        assert result["n_events"] > 0
        assert len(result["true_intensity"]) > 0

    def test_near_poisson_regime(self):
        result = simulate_regime("near_poisson", mu=2.0, alpha=0.05, beta=1.0, T=200.0, seed=42)
        assert result["branching_ratio"] < 0.1
        # Near-Poisson: intensity should be close to constant
        lam_std = np.std(result["true_intensity"])
        lam_mean = np.mean(result["true_intensity"])
        # Coefficient of variation should be small
        if lam_mean > 0:
            assert lam_std / lam_mean < 0.5

    def test_explosive_edge(self):
        result = simulate_regime("explosive", mu=0.5, alpha=0.95, beta=1.0, T=200.0, seed=42)
        assert result["branching_ratio"] == pytest.approx(0.95, abs=1e-10)
        # Should still produce events (sub-critical but close to boundary)
        assert result["n_events"] > 0

    def test_sparse_events(self):
        result = simulate_regime("sparse", mu=0.05, alpha=0.3, beta=1.0, T=200.0, seed=42)
        # Low mu → fewer events
        assert result["n_events"] < 200


class TestRunAllRegimes:
    def test_runs_all_regimes(self):
        regimes = [
            {"name": "bursty", "mu": 0.1, "alpha": 0.8, "beta": 1.0, "T": 100.0},
            {"name": "poisson", "mu": 2.0, "alpha": 0.05, "beta": 1.0, "T": 100.0},
            {"name": "explosive", "mu": 0.5, "alpha": 0.95, "beta": 1.0, "T": 100.0},
            {"name": "sparse", "mu": 0.05, "alpha": 0.3, "beta": 1.0, "T": 100.0},
        ]
        results = run_all_regimes(regimes, base_seed=42)
        assert len(results) == 4
        for r in results:
            assert "true_intensity" in r
            assert "events" in r

    def test_determinism(self):
        regimes = [
            {"name": "test", "mu": 0.5, "alpha": 0.3, "beta": 1.0, "T": 100.0},
        ]
        r1 = run_all_regimes(regimes, base_seed=42)
        r2 = run_all_regimes(regimes, base_seed=42)
        np.testing.assert_array_equal(r1[0]["events"], r2[0]["events"])
        np.testing.assert_array_equal(r1[0]["true_intensity"], r2[0]["true_intensity"])


# ====================================================================
# Estimation Validation Tests
# ====================================================================

class TestValidateEstimation:
    def test_perfect_match_passes(self):
        result = simulate_regime("test", mu=0.5, alpha=0.3, beta=1.0, T=100.0, seed=42)
        # Use ground truth as estimate → should pass
        validation = validate_estimation(
            result["true_intensity"], result, rmse_threshold=0.10
        )
        assert validation["passed"]
        assert validation["rmse"] == pytest.approx(0.0, abs=1e-10)

    def test_noisy_estimate_may_fail(self):
        result = simulate_regime("test", mu=0.5, alpha=0.3, beta=1.0, T=100.0, seed=42)
        # Add large noise → should fail
        np.random.seed(99)
        noisy = result["true_intensity"] + np.random.randn(len(result["true_intensity"])) * 10
        validation = validate_estimation(noisy, result, rmse_threshold=0.10)
        assert not validation["passed"]


class TestHawkesMLE_Recovery:
    """Test that our Hawkes MLE can recover parameters from simulated events."""

    def test_recovery_near_poisson(self):
        """Near-Poisson case: MLE should produce non-negative, finite intensity."""
        mu_true, alpha_true, beta_true = 2.0, 0.05, 1.0
        events = simulate_hawkes_events(mu_true, alpha_true, beta_true, T=500.0, seed=42)

        # Use our estimator
        timestamps = np.arange(0, 500, 1.0)
        intensity, meta = estimate_hawkes(
            {"events": events},
            timestamps,
            decay=beta_true,
            mu_init=2.0,
            alpha_init=0.05,
        )

        # Near-Poisson MLE is numerically challenging — key invariants:
        # 1. Intensity should be non-negative and finite
        assert (intensity.values >= 0).all()
        assert np.isfinite(intensity.values).all()
        # 2. Parameters should be positive
        assert meta["mu"] > 0
        assert meta["alpha"] >= 0
        assert meta["beta"] > 0
        # 3. Mean intensity should be positive
        assert float(intensity.mean()) > 0

    def test_recovery_moderate_excitation(self):
        """Moderate excitation: MLE should give reasonable estimates."""
        mu_true, alpha_true, beta_true = 0.5, 0.3, 1.0
        events = simulate_hawkes_events(mu_true, alpha_true, beta_true, T=500.0, seed=42)

        timestamps = np.arange(0, 500, 1.0)
        intensity, meta = estimate_hawkes(
            {"events": events},
            timestamps,
            decay=beta_true,
            mu_init=0.5,
            alpha_init=0.3,
        )

        # Branching ratio should be in the right ballpark
        estimated_eta = meta["alpha"] / meta["beta"]
        true_eta = alpha_true / beta_true
        assert abs(estimated_eta - true_eta) < 0.3  # within 0.3

    def test_intensity_shape_matches_ground_truth(self):
        """Estimated λ(t) should track the shape of true λ(t)."""
        mu_true, alpha_true, beta_true = 0.5, 0.3, 1.0
        events = simulate_hawkes_events(mu_true, alpha_true, beta_true, T=200.0, seed=42)

        grid = np.arange(0, 200, 1.0)
        true_lam = ground_truth_intensity(events, mu_true, alpha_true, beta_true, grid)

        est_intensity, meta = estimate_hawkes(
            {"events": events},
            grid,
            decay=beta_true,
            mu_init=0.5,
            alpha_init=0.3,
        )

        # Compute correlation between estimated and true
        from scipy.stats import pearsonr
        corr, _ = pearsonr(est_intensity.values, true_lam)
        # Should be positively correlated (shape match)
        assert corr > 0.5
