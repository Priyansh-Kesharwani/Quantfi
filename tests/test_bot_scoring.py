"""
Unit tests for bot.scoring: composite, entry, exit scores and breakdown.

Synthetic component Series; determinism with fixed seed.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bot.scoring import compute_composite_scores


def _synthetic_components(n: int = 100, seed: int = 42) -> dict:
    """Normalized [0,1] component Series for testing."""
    np.random.seed(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="min", tz="UTC")
    return {
        "T_t": pd.Series(0.5 + np.random.rand(n) * 0.1, index=idx),
        "U_t": pd.Series(0.5 + np.random.rand(n) * 0.1, index=idx),
        "O_t": pd.Series(0.5 + np.random.rand(n) * 0.1, index=idx),
        "H_t": pd.Series(0.5 + np.random.rand(n) * 0.1, index=idx),
        "LDC_t": pd.Series(0.5 + np.random.rand(n) * 0.1, index=idx),
        "C_t": pd.Series(0.8 + np.random.rand(n) * 0.2, index=idx),
        "L_t": pd.Series(0.8 + np.random.rand(n) * 0.2, index=idx),
        "R_t": pd.Series(0.5 + np.random.rand(n) * 0.5, index=idx),
        "TBL_flag": pd.Series(0.5 * np.ones(n), index=idx),
        "OFI_rev": pd.Series(0.5 * np.ones(n), index=idx),
        "lambda_decay": pd.Series(0.5 * np.ones(n), index=idx),
    }


def test_compute_composite_scores_returns_four():
    """Returns composite_score, entry_score, exit_score, breakdown."""
    comp = _synthetic_components(80, seed=1)
    composite, entry, exit_s, breakdown = compute_composite_scores(comp)
    assert composite is not None
    assert entry is not None
    assert exit_s is not None
    assert breakdown is not None
    assert len(composite) == len(entry) == len(exit_s) == len(breakdown)


def test_composite_score_in_bounds():
    """CompositeScore_t and EntryScore_t in [0, 100]."""
    comp = _synthetic_components(60, seed=2)
    composite, entry, exit_s, breakdown = compute_composite_scores(comp)
    valid = composite.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()
    valid_e = entry.dropna()
    assert (valid_e >= 0).all() and (valid_e <= 100).all()
    valid_x = exit_s.dropna()
    assert (valid_x >= 0).all() and (valid_x <= 100).all()


def test_breakdown_columns_present():
    """Breakdown contains Opp_t, Gate_t, RawFavor_t and score columns."""
    comp = _synthetic_components(50, seed=3)
    _, _, _, breakdown = compute_composite_scores(comp)
    assert "Opp_t" in breakdown.columns
    assert "Gate_t" in breakdown.columns
    assert "RawFavor_t" in breakdown.columns or "RawFavor" in breakdown.columns
    assert len(breakdown) == 50


def test_scoring_deterministic():
    """Same components and config produce identical scores (fixed seed)."""
    comp = _synthetic_components(70, seed=4)
    c1, e1, x1, b1 = compute_composite_scores(comp)
    c2, e2, x2, b2 = compute_composite_scores(comp)
    pd.testing.assert_series_equal(c1, c2)
    pd.testing.assert_series_equal(e1, e2)
    pd.testing.assert_series_equal(x1, x2)
    pd.testing.assert_frame_equal(b1, b2)


def test_scoring_accepts_config_dict():
    """Optional config dict overrides file-based config."""
    comp = _synthetic_components(40, seed=5)
    config = {
        "composite": {"trim_frac": 0.0, "r_thresh": 0.1, "S_scale": 1.0, "k_pers": 6.0},
        "exit": {"gamma1": 1.0, "gamma2": 1.0, "gamma3": 1.0},
    }
    composite, entry, exit_s, breakdown = compute_composite_scores(comp, config=config)
    assert len(composite) == 40
    assert (composite.dropna() >= 0).all() and (composite.dropna() <= 100).all()
