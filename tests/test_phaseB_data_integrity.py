"""
Phase B — Tests for Data Integrity & Validation Module.

Validates:
  - DataFrame validation contract (columns, index, NaNs, bounds)
  - Cleaning pipeline (interpolation, dedup, timezone)
  - Canonicalization pipeline
  - Structured error logging
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from validation.data_integrity import (
    validate_dataframe,
    clean_dataframe,
    canonicalize,
    fetch_and_validate,
    load_phaseB_config,
    REQUIRED_COLUMNS,
)


# ── Fixtures ──────────────────────────────────────────────

@pytest.fixture
def valid_df():
    """Create a valid OHLCV DataFrame."""
    n = 200
    dates = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.maximum(close, 1)  # ensure positive

    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + np.abs(np.random.randn(n)) * 0.5,
        "low": close - np.abs(np.random.randn(n)) * 0.5,
        "close": close,
        "volume": np.abs(np.random.randn(n) * 1e6 + 5e6),
    }, index=dates)


@pytest.fixture
def missing_cols_df(valid_df):
    """DataFrame missing required columns."""
    return valid_df.drop(columns=["volume"])


@pytest.fixture
def non_monotonic_df(valid_df):
    """DataFrame with non-monotonic index."""
    shuffled = valid_df.sample(frac=1, random_state=42)
    return shuffled


@pytest.fixture
def nan_heavy_df(valid_df):
    """DataFrame with excessive NaNs."""
    df = valid_df.copy()
    nan_mask = np.random.RandomState(42).rand(len(df)) < 0.05  # 5% NaN
    df.loc[nan_mask, "volume"] = np.nan
    return df


@pytest.fixture
def no_tz_df(valid_df):
    """DataFrame with no timezone on index."""
    df = valid_df.copy()
    df.index = df.index.tz_localize(None)
    return df


# ====================================================================
# Validation Tests
# ====================================================================

class TestValidateDataframe:
    def test_valid_df_passes(self, valid_df):
        result = validate_dataframe(valid_df, symbol="TEST")
        assert result["is_valid"] is True
        assert len(result["errors"]) == 0
        assert result["symbol"] == "TEST"

    def test_missing_columns_fails(self, missing_cols_df):
        result = validate_dataframe(missing_cols_df, symbol="TEST")
        assert result["is_valid"] is False
        assert any("Missing required columns" in e for e in result["errors"])

    def test_non_monotonic_fails(self, non_monotonic_df):
        result = validate_dataframe(non_monotonic_df, symbol="TEST")
        assert result["is_valid"] is False
        assert any("monotonic" in e.lower() for e in result["errors"])

    def test_nan_heavy_fails(self, nan_heavy_df):
        result = validate_dataframe(nan_heavy_df, symbol="TEST", max_nan_pct=0.001)
        assert result["is_valid"] is False
        assert any("NaN" in e for e in result["errors"])

    def test_no_timezone_warns(self, no_tz_df):
        result = validate_dataframe(no_tz_df, symbol="TEST")
        # Missing timezone is a warning, not an error
        assert any("timezone" in w.lower() for w in result["warnings"])

    def test_negative_prices_fail(self, valid_df):
        df = valid_df.copy()
        df.iloc[10, df.columns.get_loc("close")] = -5.0
        result = validate_dataframe(df, symbol="TEST")
        assert result["is_valid"] is False
        assert any("Negative" in e or "zero" in e for e in result["errors"])

    def test_negative_volume_fails(self, valid_df):
        df = valid_df.copy()
        df.iloc[5, df.columns.get_loc("volume")] = -100
        result = validate_dataframe(df, symbol="TEST")
        assert result["is_valid"] is False
        assert any("volume" in e.lower() for e in result["errors"])

    def test_high_lt_low_warns(self, valid_df):
        df = valid_df.copy()
        # Force high < low for one bar
        df.iloc[10, df.columns.get_loc("high")] = df.iloc[10]["low"] - 1.0
        result = validate_dataframe(df, symbol="TEST")
        assert any("High < Low" in w for w in result["warnings"])

    def test_stats_populated(self, valid_df):
        result = validate_dataframe(valid_df, symbol="TEST")
        stats = result["stats"]
        assert "n_rows" in stats
        assert stats["n_rows"] == len(valid_df)
        assert "date_start" in stats
        assert "date_end" in stats

    def test_custom_required_columns(self, valid_df):
        # Ask for a column that doesn't exist
        result = validate_dataframe(
            valid_df, symbol="TEST",
            required_columns=["open", "close", "vwap"]
        )
        assert result["is_valid"] is False
        assert any("vwap" in e for e in result["errors"])

    def test_non_datetime_index_fails(self):
        df = pd.DataFrame({
            "open": [100], "high": [101], "low": [99],
            "close": [100.5], "volume": [1e6]
        }, index=[0])
        result = validate_dataframe(df, symbol="TEST")
        assert result["is_valid"] is False
        assert any("DatetimeIndex" in e for e in result["errors"])


# ====================================================================
# Cleaning Tests
# ====================================================================

class TestCleanDataframe:
    def test_sorts_non_monotonic(self, non_monotonic_df):
        cleaned, report = clean_dataframe(non_monotonic_df)
        assert cleaned.index.is_monotonic_increasing
        assert any("sorted" in op for op in report["operations"])

    def test_interpolates_small_nans(self, valid_df):
        df = valid_df.copy()
        # Insert a tiny fraction of NaNs (within tolerance)
        df.iloc[50, df.columns.get_loc("volume")] = np.nan
        cleaned, report = clean_dataframe(df, max_nan_pct=0.01)
        assert cleaned["volume"].isna().sum() == 0

    def test_removes_duplicates(self, valid_df):
        # Duplicate one row
        df = pd.concat([valid_df, valid_df.iloc[[0]]])
        cleaned, report = clean_dataframe(df)
        assert not cleaned.index.duplicated().any()

    def test_adds_utc_timezone(self, no_tz_df):
        cleaned, report = clean_dataframe(no_tz_df)
        assert str(cleaned.index.tz) == "UTC"
        assert any("UTC" in op for op in report["operations"])

    def test_lowercases_columns(self):
        df = pd.DataFrame({
            "Open": [100], "HIGH": [101], "Low": [99],
            "CLOSE": [100.5], "Volume": [1e6]
        }, index=pd.date_range("2020-01-01", periods=1, tz="UTC"))
        cleaned, report = clean_dataframe(df)
        assert all(c.islower() for c in cleaned.columns)


# ====================================================================
# Canonicalization Tests
# ====================================================================

class TestCanonicalize:
    def test_valid_df_returns_canonical(self, valid_df):
        canonical, report = canonicalize(valid_df, symbol="TEST")
        assert isinstance(canonical, pd.DataFrame)
        assert "initial_validation" in report
        assert "cleaning" in report

    def test_dirty_df_gets_cleaned(self, non_monotonic_df):
        canonical, report = canonicalize(non_monotonic_df, symbol="TEST")
        assert canonical.index.is_monotonic_increasing
        # Should have post-clean validation
        assert "post_clean_validation" in report or "cleaning" in report

    def test_canonical_columns_lowercase(self, valid_df):
        canonical, _ = canonicalize(valid_df, symbol="TEST")
        assert all(c.islower() for c in canonical.columns)


# ====================================================================
# Edge Cases
# ====================================================================

class TestEdgeCases:
    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df.index = pd.DatetimeIndex([], tz="UTC")
        result = validate_dataframe(df, symbol="EMPTY")
        # Should not crash
        assert "n_rows" in result["stats"]
        assert result["stats"]["n_rows"] == 0

    def test_single_row(self, valid_df):
        df = valid_df.iloc[:1]
        result = validate_dataframe(df, symbol="SINGLE")
        assert result["stats"]["n_rows"] == 1

    def test_all_nan_column_fails(self, valid_df):
        df = valid_df.copy()
        df["volume"] = np.nan
        result = validate_dataframe(df, symbol="TEST", max_nan_pct=0.001)
        assert result["is_valid"] is False


# ====================================================================
# Timezone Conversion Tests
# ====================================================================

class TestTimezoneConversion:
    def test_non_utc_converted_to_utc(self):
        """clean_dataframe should convert non-UTC timezone to UTC."""
        n = 50
        dates = pd.date_range("2020-01-01", periods=n, freq="B", tz="America/New_York")
        df = pd.DataFrame({
            "open": np.random.randn(n) + 100,
            "high": np.random.randn(n) + 101,
            "low": np.random.randn(n) + 99,
            "close": np.random.randn(n) + 100,
            "volume": np.abs(np.random.randn(n) * 1e6),
        }, index=dates)
        cleaned, report = clean_dataframe(df)
        assert str(cleaned.index.tz) == "UTC"
        assert any("converted" in op or "UTC" in op for op in report["operations"])

    def test_utc_stays_utc(self):
        """clean_dataframe should leave UTC timezone as-is."""
        n = 50
        dates = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
        df = pd.DataFrame({
            "open": np.random.randn(n) + 100,
            "high": np.random.randn(n) + 101,
            "low": np.random.randn(n) + 99,
            "close": np.random.randn(n) + 100,
            "volume": np.abs(np.random.randn(n) * 1e6),
        }, index=dates)
        cleaned, report = clean_dataframe(df)
        assert str(cleaned.index.tz) == "UTC"
        # Should not have a conversion operation
        tz_ops = [op for op in report["operations"] if "converted" in op]
        assert len(tz_ops) == 0

    def test_canonicalize_produces_utc_from_non_utc(self):
        """canonicalize should produce UTC even when input is non-UTC."""
        n = 100
        dates = pd.date_range("2020-01-01", periods=n, freq="B", tz="Europe/London")
        df = pd.DataFrame({
            "Open": np.random.randn(n) + 100,
            "High": np.random.randn(n) + 101,
            "Low": np.random.randn(n) + 99,
            "Close": np.random.randn(n) + 100,
            "Volume": np.abs(np.random.randn(n) * 1e6),
        }, index=dates)
        canonical, report = canonicalize(df, symbol="TEST_TZ")
        assert str(canonical.index.tz) == "UTC"
        assert all(c.islower() for c in canonical.columns)


# ====================================================================
# Config Loader Tests
# ====================================================================

class TestLoadPhaseBConfig:
    def test_loads_config(self):
        config = load_phaseB_config("config/phaseB.yml")
        assert isinstance(config, dict)
        assert "seed" in config
        assert "data" in config
        assert "walkforward" in config
        assert "kfold" in config
        assert "hawkes_sim" in config
        assert "metrics" in config
        assert "reporting" in config

    def test_config_seed_value(self):
        config = load_phaseB_config("config/phaseB.yml")
        assert config["seed"] == 42

    def test_config_symbols(self):
        config = load_phaseB_config("config/phaseB.yml")
        symbols = config["data"]["symbols"]
        assert "SPY" in symbols
        assert "AAPL" in symbols
        assert "GLD" in symbols

    def test_config_hawkes_regimes(self):
        config = load_phaseB_config("config/phaseB.yml")
        regimes = config["hawkes_sim"]["regimes"]
        assert len(regimes) == 4
        names = [r["name"] for r in regimes]
        assert "bursty" in names
        assert "near_poisson" in names
        assert "explosive_edge" in names
        assert "sparse_events" in names

    def test_missing_config_raises(self):
        with pytest.raises(FileNotFoundError):
            load_phaseB_config("config/does_not_exist.yml")


# ====================================================================
# Walk-Forward / K-Fold Config Dataclass Tests
# ====================================================================

class TestConfigDataclasses:
    def test_walkforward_config_defaults(self):
        from validation.walkforward import WalkForwardConfig
        cfg = WalkForwardConfig()
        assert cfg.train_window == 756
        assert cfg.test_window == 252
        assert cfg.expanding is True
        assert cfg.forward_horizons == [5, 10, 20]

    def test_walkforward_config_from_dict(self):
        from validation.walkforward import WalkForwardConfig
        d = {"train_window": 500, "test_window": 100, "expanding": False}
        cfg = WalkForwardConfig.from_dict(d)
        assert cfg.train_window == 500
        assert cfg.test_window == 100
        assert cfg.expanding is False

    def test_purged_kfold_config_defaults(self):
        from validation.kfold import PurgedKFoldConfig
        cfg = PurgedKFoldConfig()
        assert cfg.n_splits == 5
        assert cfg.embargo_bars == 20

    def test_purged_kfold_config_from_dict(self):
        from validation.kfold import PurgedKFoldConfig
        d = {"n_splits": 3, "embargo_bars": 10}
        cfg = PurgedKFoldConfig.from_dict(d)
        assert cfg.n_splits == 3
        assert cfg.embargo_bars == 10

    def test_walkforward_config_from_phaseB_yml(self):
        from validation.walkforward import WalkForwardConfig
        config = load_phaseB_config("config/phaseB.yml")
        cfg = WalkForwardConfig.from_dict(config["walkforward"])
        assert cfg.train_window == 756
        assert cfg.test_window == 252
        assert cfg.forward_horizons == [5, 10, 20]

    def test_purged_kfold_config_from_phaseB_yml(self):
        from validation.kfold import PurgedKFoldConfig
        config = load_phaseB_config("config/phaseB.yml")
        cfg = PurgedKFoldConfig.from_dict(config["kfold"])
        assert cfg.n_splits == 5
        assert cfg.embargo_bars == 20
