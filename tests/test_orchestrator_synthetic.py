"""
Regression tests for orchestrator (validator + objective) on synthetic data.

Uses a stub data_loader to avoid real data fetches; verifies pipeline runs.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from engine.validation.validator import CPCVConfig, generate_cpcv_splits
from engine.validation.objective import compute_gt_score, equity_curve_from_result


def _synthetic_data_loader(config_dict):
    n = 252 * 3
    dates = pd.date_range("2019-01-01", periods=n, freq="B")
    close = 100 * np.cumprod(1 + np.random.RandomState(42).randn(n) * 0.01)
    from engine.backtester.portfolio_simulator import AssetData

    assets = {
        "SYM": AssetData(
            symbol="SYM",
            open=close,
            high=close * 1.01,
            low=close * 0.99,
            close=close,
            score=np.clip(50 + np.cumsum(np.random.RandomState(43).randn(n)) * 2, 0, 100),
            atr=close * 0.02,
            tradeable=np.ones(n, dtype=bool),
            first_valid_idx=50,
            cost_class="US_EQ_FROM_IN",
        )
    }
    return dates, assets


class TestOrchestratorWithStubData:
    def test_run_orchestrator_with_data_loader(self, tmp_path):
        import yaml
        from engine.validation.orchestrator import run_orchestrator

        config = {
            "data": {"symbols": ["SYM"]},
            "cpcv": {"n_folds": 3, "k_test": 2, "horizon_bars": 5, "embargo_bars": 3},
            "mfbo": {
                "n_trials": 2,
                "search_space": {"entry_score_threshold": [70, 80], "max_positions": [5, 10]},
                "fidelity_spec": {"low": {"cost": 1.0}},
            },
            "dsr_min": 0.0,
        }
        config_path = tmp_path / "tuning.yml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        result = run_orchestrator(str(config_path), tmp_path / "out", 42, data_loader=_synthetic_data_loader)
        assert result.artifact_paths
        assert (tmp_path / "out" / "winning_config.json").is_file() or result.winning_config is None
