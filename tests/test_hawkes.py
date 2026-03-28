"""Phase A — Hawkes intensity λ(t) unit tests."""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.indicators.hawkes import (
    estimate_hawkes,
    hawkes_lambda_decay,
    _hawkes_log_likelihood,
    _fit_hawkes_mle,
    _compute_intensity,
)
from tests.fixtures import poisson_events, hawkes_events

class TestHawkesLogLikelihood:

    def test_empty_events_returns_mu_T(self):
        """With no events, NLL = μ·T."""
        nll = _hawkes_log_likelihood(
            np.array([0.5, 0.3, 1.0]),
            event_times=np.array([]),
            T_end=100.0,
        )
        assert abs(nll - 0.5 * 100.0) < 1e-6

    def test_infeasible_params_return_large(self):
        nll = _hawkes_log_likelihood(
            np.array([-1.0, 0.3, 1.0]),
            event_times=np.array([1.0, 2.0]),
            T_end=10.0,
        )
        assert nll >= 1e10

class TestFitHawkesMLE:

    def test_recovers_poisson_baseline(self):
        """For a homogeneous Poisson process, α should be ≈ 0."""
        events = poisson_events(rate=2.0, T=200.0, seed=42)
        mu, alpha, beta = _fit_hawkes_mle(events, T_end=200.0)
        assert abs(mu - 2.0) < 1.0, f"Expected mu ≈ 2.0, got {mu:.3f}"
        assert alpha < 0.5, f"Expected small alpha, got {alpha:.3f}"

    def test_recovers_excitation(self):
        """For a self-exciting process, α should be meaningfully > 0."""
        events = hawkes_events(mu=0.5, alpha=0.8, beta=1.5, T=300.0, seed=42)
        mu, alpha, beta = _fit_hawkes_mle(events, T_end=300.0, beta_init=1.5)
        assert alpha > 0.1, f"Expected meaningful excitation, got α = {alpha:.3f}"

class TestComputeIntensity:

    def test_baseline_when_no_events(self):
        """With no events, λ(t) = μ everywhere."""
        ts = np.linspace(0, 10, 50)
        lam = _compute_intensity(np.array([]), ts, mu=0.5, alpha=0.3, beta=1.0)
        np.testing.assert_array_almost_equal(lam, 0.5)

    def test_intensity_spikes_after_event(self):
        """λ should be higher right after an event than before."""
        events = np.array([5.0])
        ts = np.array([4.9, 5.1])
        lam = _compute_intensity(events, ts, mu=0.5, alpha=2.0, beta=1.0)
        assert lam[1] > lam[0], "Intensity should spike after event"

    def test_intensity_decays_after_event(self):
        """λ should decay as we move further from the event."""
        events = np.array([5.0])
        ts = np.array([5.1, 6.0, 10.0])
        lam = _compute_intensity(events, ts, mu=0.5, alpha=2.0, beta=1.0)
        assert lam[0] > lam[1] > lam[2], "Intensity should decay over time"

class TestEstimateHawkes:

    def test_returns_series_and_meta(self):
        events = poisson_events(rate=1.0, T=50.0, seed=42)
        ts = np.arange(0, 50, 1.0)
        intensity, meta = estimate_hawkes({"trades": events}, ts)
        assert isinstance(intensity, pd.Series)
        assert len(intensity) == len(ts)
        assert "mu" in meta and "alpha" in meta and "beta" in meta

    def test_intensity_all_positive(self):
        events = hawkes_events(mu=0.5, alpha=0.3, beta=1.0, T=100.0, seed=42)
        ts = np.arange(0, 100, 1.0)
        intensity, _ = estimate_hawkes({"trades": events}, ts)
        assert (intensity >= 0).all()

    def test_deterministic(self):
        events = hawkes_events(mu=0.5, alpha=0.3, beta=1.0, T=50.0, seed=42)
        ts = np.arange(0, 50, 1.0)
        i1, _ = estimate_hawkes({"trades": events}, ts)
        i2, _ = estimate_hawkes({"trades": events}, ts)
        pd.testing.assert_series_equal(i1, i2)

    def test_meta_backend_recorded(self):
        events = hawkes_events(mu=0.5, alpha=0.3, beta=1.0, T=50.0, seed=42)
        ts = np.arange(0, 50, 1.0)
        _, meta = estimate_hawkes({"trades": events}, ts)
        assert meta["backend"] in ("tick", "custom_mle")

class TestHawkesLambdaDecay:

    def test_output_bounded_zero_one(self):
        events = hawkes_events(mu=0.5, alpha=0.3, beta=1.0, T=200.0, seed=42)
        ts = np.arange(0, 200, 1.0)
        decay = hawkes_lambda_decay({"trades": events}, ts, min_obs=50)
        valid = decay.dropna()
        assert len(valid) > 0
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_warmup_is_nan(self):
        """First min_obs observations should be NaN (ECDF warm-up)."""
        events = hawkes_events(mu=0.5, alpha=0.3, beta=1.0, T=200.0, seed=42)
        ts = np.arange(0, 200, 1.0)
        min_obs = 50
        decay = hawkes_lambda_decay({"trades": events}, ts, min_obs=min_obs)
        assert decay.iloc[:min_obs].isna().all()

    def test_constant_intensity_all_nan_short(self):
        """With no events (constant μ), short series → all NaN (warm-up)."""
        ts = np.arange(0, 10, 1.0)
        decay = hawkes_lambda_decay({"trades": np.array([])}, ts, min_obs=50)
        assert decay.isna().all()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
