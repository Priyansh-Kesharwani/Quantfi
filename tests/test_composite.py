"""Phase A — Entry/Exit composite score (compose_scores) tests."""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.indicators.composite import compose_scores, PhaseAConfig

def _make_components(n: int = 200, seed: int = 42) -> dict:
    """Generate deterministic neutral-ish component series."""
    np.random.seed(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    return {
        "T_t": pd.Series(np.random.uniform(0.3, 0.7, n), index=idx),
        "U_t": pd.Series(np.random.uniform(0.3, 0.7, n), index=idx),
        "H_t": pd.Series(np.random.uniform(0.4, 0.6, n), index=idx),
        "R_t": pd.Series(np.random.uniform(0.4, 0.8, n), index=idx),
        "C_t": pd.Series(np.random.uniform(0.5, 0.9, n), index=idx),
        "OFI_t": pd.Series(np.random.uniform(0.3, 0.7, n), index=idx),
        "P_move_t": pd.Series(np.random.uniform(0.3, 0.7, n), index=idx),
        "S_t": pd.Series(np.zeros(n), index=idx),
        "TBL_flag": pd.Series(np.random.uniform(0.2, 0.8, n), index=idx),
        "OFI_rev": pd.Series(np.random.uniform(0.2, 0.8, n), index=idx),
        "lambda_decay": pd.Series(np.random.uniform(0.2, 0.8, n), index=idx),
    }

class TestComposeScoresContract:

    def test_returns_three_outputs(self):
        comps = _make_components()
        entry, exit_, breakdown = compose_scores(comps)
        assert isinstance(entry, pd.Series)
        assert isinstance(exit_, pd.Series)
        assert isinstance(breakdown, pd.DataFrame)

    def test_entry_in_zero_hundred(self):
        comps = _make_components()
        entry, _, _ = compose_scores(comps)
        assert (entry >= 0).all() and (entry <= 100).all()

    def test_exit_in_zero_hundred(self):
        comps = _make_components()
        _, exit_, _ = compose_scores(comps)
        assert (exit_ >= 0).all() and (exit_ <= 100).all()

    def test_output_length_matches(self):
        n = 150
        comps = _make_components(n=n)
        entry, exit_, breakdown = compose_scores(comps)
        assert len(entry) == n
        assert len(exit_) == n
        assert len(breakdown) == n

    def test_breakdown_contains_key_columns(self):
        comps = _make_components()
        _, _, breakdown = compose_scores(comps)
        for col in ["Opp_t", "Gate_t", "RawFavor", "Entry_Score", "Exit_Score"]:
            assert col in breakdown.columns, f"Missing column: {col}"

class TestEntryScore:

    def test_all_favorable_high_score(self):
        n = 100
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        comps = {
            "T_t": pd.Series(np.full(n, 0.9), index=idx),
            "U_t": pd.Series(np.full(n, 0.9), index=idx),
            "H_t": pd.Series(np.full(n, 0.7), index=idx),
            "R_t": pd.Series(np.full(n, 0.9), index=idx),
            "C_t": pd.Series(np.full(n, 0.9), index=idx),
            "OFI_t": pd.Series(np.full(n, 0.9), index=idx),
            "P_move_t": pd.Series(np.full(n, 0.9), index=idx),
            "S_t": pd.Series(np.zeros(n), index=idx),
        }
        entry, _, _ = compose_scores(comps)
        assert entry.mean() > 60, f"Expected high entry score, got mean={entry.mean():.1f}"

    def test_all_unfavorable_low_score(self):
        n = 100
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        comps = {
            "T_t": pd.Series(np.full(n, 0.1), index=idx),
            "U_t": pd.Series(np.full(n, 0.1), index=idx),
            "H_t": pd.Series(np.full(n, 0.3), index=idx),
            "R_t": pd.Series(np.full(n, 0.1), index=idx),
            "C_t": pd.Series(np.full(n, 0.1), index=idx),
            "OFI_t": pd.Series(np.full(n, 0.1), index=idx),
            "P_move_t": pd.Series(np.full(n, 0.1), index=idx),
            "S_t": pd.Series(np.zeros(n), index=idx),
        }
        entry, _, _ = compose_scores(comps)
        assert entry.mean() < 40, f"Expected low entry score, got mean={entry.mean():.1f}"

    def test_S_scale_affects_spread(self):
        comps = _make_components(seed=77)
        cfg_narrow = PhaseAConfig(S_scale=0.5)
        cfg_wide = PhaseAConfig(S_scale=2.0)
        e_narrow, _, _ = compose_scores(comps, cfg_narrow)
        e_wide, _, _ = compose_scores(comps, cfg_wide)
        assert e_wide.std() > e_narrow.std()

class TestExitScore:

    def test_weights_sum_to_one(self):
        cfg = PhaseAConfig()
        assert abs(cfg.gamma_1 + cfg.gamma_2 + cfg.gamma_3 - 1.0) < 1e-9

    def test_high_tbl_flag_high_exit(self):
        n = 100
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        comps = {
            "TBL_flag": pd.Series(np.full(n, 0.95), index=idx),
            "OFI_rev": pd.Series(np.full(n, 0.95), index=idx),
            "lambda_decay": pd.Series(np.full(n, 0.95), index=idx),
        }
        _, exit_, _ = compose_scores(comps)
        assert exit_.mean() > 60

class TestPhaseAConfig:

    def test_from_yaml(self, tmp_path):
        yaml_content = """
composite:
  entry:
    committee_method: "mean"
    S_scale: 1.5
    regime_threshold: 0.4
    g_pers_type: "sigmoid"
    g_pers_params:
      k: 8.0
  exit:
    gamma_1: 0.3
    gamma_2: 0.4
    gamma_3: 0.3
    S_scale_exit: 1.2
logging:
  log_intermediates: false
"""
        p = tmp_path / "test_phaseA.yml"
        p.write_text(yaml_content)
        cfg = PhaseAConfig.from_yaml(str(p))
        assert cfg.committee_method == "mean"
        assert cfg.S_scale == 1.5
        assert cfg.gamma_2 == 0.4
        assert cfg.log_intermediates is False

    def test_to_dict(self):
        cfg = PhaseAConfig(S_scale=2.0, gamma_1=0.5)
        d = cfg.to_dict()
        assert d["S_scale"] == 2.0
        assert d["gamma_1"] == 0.5

class TestDeterminism:

    def test_same_input_same_output(self):
        comps = _make_components(seed=42)
        e1, x1, b1 = compose_scores(comps)
        e2, x2, b2 = compose_scores(comps)
        pd.testing.assert_series_equal(e1, e2)
        pd.testing.assert_series_equal(x1, x2)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
