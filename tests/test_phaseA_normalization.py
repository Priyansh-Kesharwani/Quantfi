"""Phase A — Normalization pipeline (expanding_ecdf_sigmoid) tests."""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from indicators.normalization import expanding_ecdf_sigmoid
from tests.fixtures import fbm_series, ou_series


class TestExpandingEcdfSigmoid:

    @pytest.fixture
    def fbm_close(self):
        df = fbm_series(n=400, H=0.7, seed=42)
        return df["close"]

    @pytest.fixture
    def ou_close(self):
        df = ou_series(n=400, seed=42)
        return df["close"]

    # ── Basic contract ────────────────────────────────────
    def test_returns_series(self, fbm_close):
        result = expanding_ecdf_sigmoid(fbm_close, min_obs=50)
        assert isinstance(result, pd.Series)
        assert len(result) == len(fbm_close)

    def test_warmup_is_nan(self, fbm_close):
        min_obs = 80
        result = expanding_ecdf_sigmoid(fbm_close, min_obs=min_obs)
        assert result.iloc[:min_obs].isna().all()

    def test_valid_range_zero_one(self, fbm_close):
        result = expanding_ecdf_sigmoid(fbm_close, min_obs=50)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    # ── Determinism ───────────────────────────────────────
    def test_deterministic(self, fbm_close):
        r1 = expanding_ecdf_sigmoid(fbm_close, k=1.0, polarity=1, min_obs=50)
        r2 = expanding_ecdf_sigmoid(fbm_close, k=1.0, polarity=1, min_obs=50)
        pd.testing.assert_series_equal(r1, r2)

    # ── Polarity ──────────────────────────────────────────
    def test_polarity_inversion(self, fbm_close):
        pos = expanding_ecdf_sigmoid(fbm_close, polarity=1, min_obs=50)
        neg = expanding_ecdf_sigmoid(fbm_close, polarity=-1, min_obs=50)
        valid_mask = ~(pos.isna() | neg.isna())
        # Sum should be ≈ 1 for each pair
        sums = pos[valid_mask] + neg[valid_mask]
        np.testing.assert_array_almost_equal(sums.values, 1.0, decimal=10)

    # ── Sigmoid steepness ─────────────────────────────────
    def test_higher_k_more_extreme(self, fbm_close):
        r_k1 = expanding_ecdf_sigmoid(fbm_close, k=0.5, min_obs=50)
        r_k3 = expanding_ecdf_sigmoid(fbm_close, k=3.0, min_obs=50)
        valid = ~(r_k1.isna() | r_k3.isna())
        # Higher k pushes values further from 0.5
        dev_k1 = (r_k1[valid] - 0.5).abs().mean()
        dev_k3 = (r_k3[valid] - 0.5).abs().mean()
        assert dev_k3 > dev_k1

    # ── Works with OU series ──────────────────────────────
    def test_ou_series(self, ou_close):
        result = expanding_ecdf_sigmoid(ou_close, min_obs=50)
        valid = result.dropna()
        assert len(valid) > 0
        assert (valid >= 0).all() and (valid <= 1).all()

    # ── Works with plain numpy ────────────────────────────
    def test_accepts_numpy_array(self):
        np.random.seed(99)
        arr = np.random.randn(300)
        result = expanding_ecdf_sigmoid(arr, min_obs=50)
        assert isinstance(result, pd.Series)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 1).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
