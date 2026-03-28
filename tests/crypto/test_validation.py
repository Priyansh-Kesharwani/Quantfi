"""Tests for crypto.validation.orchestrator."""

import pytest

optuna = pytest.importorskip("optuna")

from engine.crypto.validation.orchestrator import (
    CryptoOrchestrator,
    OrchestratorResult,
    sensitivity_analysis,
)


def _mock_evaluate(config):
    """Simple mock: score increases with entry_threshold, decreases with leverage."""
    base = 1.0
    base += config.get("entry_threshold", 40) / 100
    base -= config.get("leverage", 3) / 20
    base += config.get("kelly_fraction", 0.25) * 2
    return {"gt_score": base, "sharpe": base * 0.5, "dsr": base * 0.3}


class TestCryptoOrchestrator:
    def test_runs_to_completion(self):
        orch = CryptoOrchestrator(
            evaluate_fn=_mock_evaluate,
            n_trials=5,
            seed=42,
        )
        result = orch.run()
        assert isinstance(result, OrchestratorResult)
        assert result.n_trials == 5
        assert result.winning_config is not None

    def test_finds_better_config(self):
        orch = CryptoOrchestrator(
            evaluate_fn=_mock_evaluate,
            n_trials=10,
            seed=42,
        )
        result = orch.run()
        assert result.best_gt_score > 0

    def test_trial_log_populated(self):
        orch = CryptoOrchestrator(
            evaluate_fn=_mock_evaluate,
            n_trials=5,
            seed=42,
        )
        result = orch.run()
        assert len(result.trial_log) == 5

    def test_robustness_included(self):
        orch = CryptoOrchestrator(
            evaluate_fn=_mock_evaluate,
            n_trials=3,
            seed=42,
        )
        result = orch.run()
        assert "is_robust" in result.robustness


class TestSensitivityAnalysis:
    def test_detects_robust_config(self):
        config = {"entry_threshold": 50.0, "leverage": 3.0, "kelly_fraction": 0.25}
        result = sensitivity_analysis(config, 1.5, _mock_evaluate)
        assert "results" in result
        assert "is_robust" in result

    def test_detects_fragile_param(self):
        def fragile_eval(cfg):
            val = cfg.get("magic_param", 1.0)
            if abs(val - 1.0) > 0.05:
                return {"gt_score": -100}
            return {"gt_score": 10.0}

        config = {"magic_param": 1.0}
        result = sensitivity_analysis(config, 10.0, fragile_eval)
        assert not result["is_robust"]
        assert len(result["fragile_params"]) > 0

    def test_skips_zero_values(self):
        config = {"param_a": 0, "param_b": 5.0}
        result = sensitivity_analysis(config, 1.0, _mock_evaluate)
        keys = list(result["results"].keys())
        assert not any("param_a" in k for k in keys)
