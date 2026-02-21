"""Phase A — OFI unit tests."""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from indicators.ofi import compute_ofi, compute_ofi_reversal, _signed_volume
from tests.fixtures import fbm_series, ou_series


# ── Signed volume logic ────────────────────────────────────
class TestSignedVolume:

    def test_positive_move_positive_sign(self):
        close = np.array([100, 101, 102, 103, 104], dtype=float)
        volume = np.array([1e6] * 5, dtype=float)
        sv = _signed_volume(close, volume)
        # All moves are positive → signed volume should be positive (after first)
        assert np.all(sv[1:] > 0)

    def test_negative_move_negative_sign(self):
        close = np.array([104, 103, 102, 101, 100], dtype=float)
        volume = np.array([1e6] * 5, dtype=float)
        sv = _signed_volume(close, volume)
        assert np.all(sv[1:] < 0)

    def test_flat_move_zero_sign(self):
        close = np.array([100, 100, 100], dtype=float)
        volume = np.array([1e6] * 3, dtype=float)
        sv = _signed_volume(close, volume)
        assert np.all(sv == 0)

    def test_shape_matches_input(self):
        close = np.random.randn(50).cumsum() + 100
        volume = np.abs(np.random.randn(50)) * 1e6
        sv = _signed_volume(close, volume)
        assert sv.shape == close.shape


# ── OFI computation ────────────────────────────────────────
class TestComputeOFI:

    @pytest.fixture
    def sample_df(self):
        return fbm_series(n=300, seed=42)

    def test_output_length_matches_input(self, sample_df):
        ofi = compute_ofi(sample_df, window=20, normalize=False)
        assert len(ofi) == len(sample_df)

    def test_raw_ofi_not_bounded(self, sample_df):
        """Raw (un-normalised) OFI can be any real number."""
        ofi = compute_ofi(sample_df, window=20, normalize=False)
        valid = ofi.dropna()
        # Should have *some* non-zero values
        assert (valid != 0).any()

    def test_normalised_ofi_bounded_zero_one(self, sample_df):
        ofi = compute_ofi(sample_df, window=20, normalize=True, min_obs=50)
        valid = ofi.dropna()
        assert np.all(valid >= 0) and np.all(valid <= 1)

    def test_deterministic(self, sample_df):
        ofi1 = compute_ofi(sample_df, window=20, normalize=True, min_obs=50)
        ofi2 = compute_ofi(sample_df, window=20, normalize=True, min_obs=50)
        pd.testing.assert_series_equal(ofi1, ofi2)

    def test_window_affects_nan_count(self, sample_df):
        """Longer window should produce more leading NaNs."""
        ofi_short = compute_ofi(sample_df, window=5, normalize=False)
        ofi_long = compute_ofi(sample_df, window=40, normalize=False)
        nan_short = ofi_short.isna().sum()
        nan_long = ofi_long.isna().sum()
        assert nan_long > nan_short

    def test_case_insensitive_columns(self):
        """Should work with both 'close'/'Close' column names."""
        df = fbm_series(n=200, seed=99)
        df.columns = [c.capitalize() for c in df.columns]
        ofi = compute_ofi(df, window=10, normalize=False)
        assert len(ofi) == 200


class TestOFIReversal:

    def test_reversal_is_inverted_polarity(self):
        df = fbm_series(n=300, seed=42)
        ofi_normal = compute_ofi(df, window=20, normalize=True, min_obs=50)
        ofi_rev = compute_ofi_reversal(df, window=20, min_obs=50)

        # The reversal should have *opposite* polarity —
        # when normal OFI is high, reversal should be low (on average)
        valid_mask = ~(ofi_normal.isna() | ofi_rev.isna())
        corr = np.corrcoef(
            ofi_normal[valid_mask].values,
            ofi_rev[valid_mask].values,
        )[0, 1]
        assert corr < 0, f"Expected negative correlation, got {corr:.3f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
