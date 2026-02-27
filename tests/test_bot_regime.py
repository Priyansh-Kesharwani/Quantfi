"""
Unit tests for bot.regime: fit_hmm and predict_state_prob.

Synthetic returns (OU/FBM); determinism with fixed seed.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("hmmlearn")

from bot.regime import fit_hmm, predict_state_prob, regime_probability_rolling


def _synthetic_returns(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Returns series as DataFrame (one column)."""
    np.random.seed(seed)
    r = np.random.randn(n) * 0.01
    idx = pd.date_range("2020-01-01", periods=n, freq="min", tz="UTC")
    return pd.DataFrame({"returns": r}, index=idx)


def test_fit_hmm_returns_model():
    """fit_hmm returns a fitted model object."""
    df = _synthetic_returns(150, seed=1)
    model = fit_hmm(df, n_states=3, random_state=42)
    assert model is not None
    assert hasattr(model, "predict_proba")
    assert getattr(model, "n_components", None) == 3 or model.n_components == 3


def test_predict_state_prob_shape_and_sum():
    """predict_state_prob returns DataFrame; rows sum to 1 where valid."""
    df = _synthetic_returns(120, seed=2)
    model = fit_hmm(df, n_states=3, random_state=43)
    probs = predict_state_prob(df, model)
    assert isinstance(probs, pd.DataFrame)
    assert probs.index.equals(df.index)
    assert probs.shape[1] == 3
    row_sums = probs.sum(axis=1)
    valid = row_sums.notna()
    assert np.allclose(row_sums[valid].values, 1.0, rtol=1e-5)


def test_predict_state_prob_columns():
    """Output has state_0, state_1, state_2 (for n_states=3)."""
    df = _synthetic_returns(100, seed=3)
    model = fit_hmm(df, n_states=3, random_state=44)
    probs = predict_state_prob(df, model)
    for i in range(3):
        assert f"state_{i}" in probs.columns


def test_fit_hmm_deterministic():
    """Same data and random_state produce identical model params (via predict_proba)."""
    df = _synthetic_returns(180, seed=4)
    m1 = fit_hmm(df, n_states=3, random_state=99)
    m2 = fit_hmm(df, n_states=3, random_state=99)
    p1 = predict_state_prob(df, m1)
    p2 = predict_state_prob(df, m2)
    pd.testing.assert_frame_equal(p1, p2)


def test_fit_hmm_insufficient_data_raises():
    """Too few observations raises."""
    df = _synthetic_returns(5, seed=5)
    with pytest.raises(ValueError, match="Not enough"):
        fit_hmm(df, n_states=3)


def test_predict_state_prob_returns_column():
    """Can specify returns column by name."""
    np.random.seed(6)
    idx = pd.date_range("2020-01-01", periods=80, freq="min", tz="UTC")
    df = pd.DataFrame({"other": np.zeros(80), "ret": np.random.randn(80) * 0.01}, index=idx)
    model = fit_hmm(df[["ret"]], n_states=2, random_state=45)
    probs = predict_state_prob(df, model, returns_column="ret")
    assert probs.shape[0] == 80
    assert probs.shape[1] == 2


def test_regime_probability_rolling_past_only():
    df = _synthetic_returns(120, seed=7)
    probs = regime_probability_rolling(df, n_states=2, window=60, refit_every=30, random_state=99)
    assert isinstance(probs, pd.DataFrame)
    assert probs.shape[0] == 120
    assert probs.shape[1] == 2
    valid = probs.notna().all(axis=1)
    assert valid.sum() > 0
    assert np.allclose(probs.loc[valid].sum(axis=1).values, 1.0, rtol=1e-5)
