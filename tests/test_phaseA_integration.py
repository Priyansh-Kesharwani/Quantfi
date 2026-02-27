"""Phase A — End-to-end integration tests.

Runs the full Phase A pipeline (indicators → normalisation → composite)
with synthetic data and verifies contracts, determinism, and cross-indicator
consistency.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from indicators.normalization import expanding_ecdf_sigmoid
from indicators.ofi import compute_ofi, compute_ofi_reversal
from indicators.hawkes import estimate_hawkes, hawkes_lambda_decay
from indicators.ldc import LDC, build_templates_from_labels
from indicators.composite import compose_scores, PhaseAConfig
from tests.fixtures import fbm_series, hawkes_events

SEED = 42
MIN_OBS = 50

def _run_full_pipeline(seed: int = SEED):
    """Execute the complete Phase A pipeline and return all artefacts."""
    df = fbm_series(n=400, H=0.7, seed=seed)

    ofi = compute_ofi(df, window=20, normalize=True, min_obs=MIN_OBS)
    ofi_rev = compute_ofi_reversal(df, window=20, min_obs=MIN_OBS)

    h_events = hawkes_events(mu=0.5, alpha=0.3, beta=1.0, T=200.0, seed=seed)
    timestamps = np.arange(0, 200, 1.0)
    intensity, hawkes_meta = estimate_hawkes(
        {"trades": h_events}, timestamps, decay=1.0,
    )
    lam_decay = hawkes_lambda_decay(
        {"trades": h_events}, timestamps, min_obs=40,
    )

    np.random.seed(seed)
    n_feat = 5
    bull_feat = np.random.randn(50, n_feat) + 1.0
    bear_feat = np.random.randn(50, n_feat) - 1.0
    features = np.vstack([bull_feat, bear_feat])
    labels = np.array([1] * 50 + [0] * 50)
    templates = build_templates_from_labels(features, labels)
    ldc = LDC(kappa=1.0)
    ldc.fit(templates)
    ldc_scores = ldc.score_batch(features)

    n = len(df)
    idx = df.index
    np.random.seed(seed)
    components = {
        "T_t": pd.Series(np.random.uniform(0.3, 0.7, n), index=idx),
        "U_t": pd.Series(np.random.uniform(0.3, 0.7, n), index=idx),
        "H_t": pd.Series(np.random.uniform(0.4, 0.6, n), index=idx),
        "R_t": pd.Series(np.random.uniform(0.4, 0.8, n), index=idx),
        "C_t": pd.Series(np.random.uniform(0.5, 0.9, n), index=idx),
        "OFI_t": ofi,
        "P_move_t": pd.Series(np.random.uniform(0.3, 0.7, n), index=idx),
        "S_t": pd.Series(np.zeros(n), index=idx),
        "TBL_flag": pd.Series(np.random.uniform(0.2, 0.8, n), index=idx),
        "OFI_rev": ofi_rev,
        "lambda_decay": pd.Series(np.random.uniform(0.2, 0.8, n), index=idx),
    }
    entry, exit_, breakdown = compose_scores(components)

    return {
        "df": df,
        "ofi": ofi,
        "ofi_rev": ofi_rev,
        "intensity": intensity,
        "lam_decay": lam_decay,
        "ldc_scores": ldc_scores,
        "entry": entry,
        "exit": exit_,
        "breakdown": breakdown,
        "components": components,
    }

class TestPipelineDeterminism:
    """Identical seeds → bit-for-bit identical outputs."""

    def test_two_runs_match(self):
        r1 = _run_full_pipeline(seed=42)
        r2 = _run_full_pipeline(seed=42)

        pd.testing.assert_series_equal(r1["ofi"], r2["ofi"])
        pd.testing.assert_series_equal(r1["ofi_rev"], r2["ofi_rev"])
        pd.testing.assert_series_equal(r1["intensity"], r2["intensity"])
        pd.testing.assert_series_equal(r1["lam_decay"], r2["lam_decay"])
        np.testing.assert_array_equal(r1["ldc_scores"], r2["ldc_scores"])
        pd.testing.assert_series_equal(r1["entry"], r2["entry"])
        pd.testing.assert_series_equal(r1["exit"], r2["exit"])
        pd.testing.assert_frame_equal(r1["breakdown"], r2["breakdown"])

class TestIndicatorContracts:
    """All normalised series satisfy ∈ [0, 1] (ignoring warm-up NaN)."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        return _run_full_pipeline()

    def test_ofi_bounded(self, pipeline):
        valid = pipeline["ofi"].dropna()
        assert len(valid) > 0
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_ofi_rev_bounded(self, pipeline):
        valid = pipeline["ofi_rev"].dropna()
        assert len(valid) > 0
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_hawkes_intensity_positive(self, pipeline):
        valid = pipeline["intensity"].dropna()
        assert (valid >= 0).all()

    def test_hawkes_decay_bounded(self, pipeline):
        valid = pipeline["lam_decay"].dropna()
        assert len(valid) > 0
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_ldc_scores_bounded(self, pipeline):
        scores = pipeline["ldc_scores"]
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_entry_score_in_zero_hundred(self, pipeline):
        entry = pipeline["entry"]
        assert (entry >= 0).all() and (entry <= 100).all()

    def test_exit_score_in_zero_hundred(self, pipeline):
        exit_ = pipeline["exit"].dropna()
        assert len(exit_) > 0
        assert (exit_ >= 0).all() and (exit_ <= 100).all()

class TestCrossIndicatorConsistency:
    """Verify relationships between indicators make sense."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        return _run_full_pipeline()

    def test_ofi_and_ofi_rev_sum_near_one(self, pipeline):
        """OFI + OFI_rev should be close to 1 where both are valid."""
        ofi = pipeline["ofi"]
        ofi_rev = pipeline["ofi_rev"]
        total = ofi + ofi_rev
        valid = total.dropna()
        assert np.abs(valid.mean() - 1.0) < 0.15, (
            f"OFI + OFI_rev mean should be ~1.0, got {valid.mean():.3f}"
        )

    def test_breakdown_entry_exit_present(self, pipeline):
        bd = pipeline["breakdown"]
        assert "Entry_Score" in bd.columns
        assert "Exit_Score" in bd.columns

    def test_entry_matches_breakdown(self, pipeline):
        entry = pipeline["entry"]
        bd_entry = pipeline["breakdown"]["Entry_Score"]
        pd.testing.assert_series_equal(entry, bd_entry, check_names=False)

    def test_exit_matches_breakdown(self, pipeline):
        exit_ = pipeline["exit"]
        bd_exit = pipeline["breakdown"]["Exit_Score"]
        pd.testing.assert_series_equal(exit_, bd_exit, check_names=False)

class TestNormalisationPipelineIntegrity:
    """Verify the expanding ECDF-sigmoid pipeline properties in context."""

    def test_warm_up_nans_present(self):
        """Indicators should have NaN warm-up period from ECDF."""
        r = _run_full_pipeline()
        assert r["ofi"].iloc[:MIN_OBS].isna().any(), (
            "OFI should have NaN in warm-up region"
        )

    def test_no_lookahead_in_ofi(self):
        """Appending new rows shouldn't change past OFI values.

        We generate a single FBM series, compute OFI on [0:200],
        then compute OFI on [0:300] and verify the first 200 match.
        """
        df_full = fbm_series(n=300, H=0.7, seed=42)
        df_short = df_full.iloc[:200].copy()

        ofi_short = compute_ofi(df_short, window=20, normalize=True, min_obs=MIN_OBS)
        ofi_long = compute_ofi(df_full, window=20, normalize=True, min_obs=MIN_OBS)

        pd.testing.assert_series_equal(
            ofi_short,
            ofi_long.iloc[:200],
            check_names=False,
        )

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
