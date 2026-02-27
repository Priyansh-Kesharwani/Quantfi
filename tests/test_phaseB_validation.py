"""
Phase B — Tests for Validation Engine (Walk-Forward + Purged K-Fold + Metrics).

Uses synthetic data from Phase A fixtures to verify:
  - Metric calculations (IC, hit rate, Sortino, CAGR, drawdown, signals)
  - Walk-forward fold generation and metric aggregation
  - Purged K-fold splitting with embargo zones
  - Report generator rendering
  - Deterministic reproducibility
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.fixtures import fbm_series, ou_series
from validation.metrics import (
    information_coefficient,
    hit_rate,
    sortino_ratio,
    max_drawdown,
    cagr,
    forward_returns,
    evaluate_signals,
    compute_score_metrics,
    compute_all_metrics,
)
from validation.walkforward import (
    walkforward_cv,
    _generate_folds,
    WalkForwardResult,
)
from validation.kfold import (
    purged_kfold,
    _purged_kfold_splits,
    KFoldResult,
)

@pytest.fixture(scope="module")
def synth_df():
    """Synthetic price DataFrame (FBM-based, 1000 bars)."""
    return fbm_series(n=1000, H=0.6, seed=42)

@pytest.fixture(scope="module")
def synth_scores(synth_df):
    """Deterministic synthetic Entry_Score and Exit_Score."""
    np.random.seed(42)
    n = len(synth_df)
    close = synth_df["close"].values
    fwd_5 = np.roll(close, -5) / close - 1
    fwd_5[-5:] = 0

    signal = (fwd_5 - fwd_5.mean()) / max(fwd_5.std(), 1e-12)
    entry = 50 + 15 * signal + np.random.randn(n) * 10
    entry = np.clip(entry, 0, 100)
    exit_ = 100 - entry + np.random.randn(n) * 5
    exit_ = np.clip(exit_, 0, 100)

    return (
        pd.Series(entry, index=synth_df.index, name="Entry_Score"),
        pd.Series(exit_, index=synth_df.index, name="Exit_Score"),
    )

def _dummy_score_fn(df):
    """A deterministic score function for testing walk-forward / kfold."""
    np.random.seed(42)
    n = len(df)
    entry = pd.Series(
        np.clip(50 + np.random.randn(n) * 15, 0, 100),
        index=df.index, name="Entry_Score"
    )
    exit_ = pd.Series(
        np.clip(50 + np.random.randn(n) * 15, 0, 100),
        index=df.index, name="Exit_Score"
    )
    return entry, exit_

class TestInformationCoefficient:
    def test_perfect_positive_ic(self):
        scores = pd.Series([10, 20, 30, 40, 50])
        returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
        ic = information_coefficient(scores, returns)
        assert ic == pytest.approx(1.0, abs=1e-6)

    def test_perfect_negative_ic(self):
        scores = pd.Series([50, 40, 30, 20, 10])
        returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
        ic = information_coefficient(scores, returns)
        assert ic == pytest.approx(-1.0, abs=1e-6)

    def test_nan_handling(self):
        scores = pd.Series([10, np.nan, 30, 40, 50])
        returns = pd.Series([0.01, 0.02, np.nan, 0.04, 0.05])
        ic = information_coefficient(scores, returns)
        assert not np.isnan(ic)

    def test_too_few_samples_returns_nan(self):
        scores = pd.Series([10, 20])
        returns = pd.Series([np.nan, np.nan])
        ic = information_coefficient(scores, returns)
        assert np.isnan(ic)

class TestHitRate:
    def test_all_hits(self):
        scores = pd.Series([80, 90, 75, 85])
        returns = pd.Series([0.01, 0.02, 0.01, 0.03])
        hr = hit_rate(scores, returns, threshold=70)
        assert hr == 1.0

    def test_no_hits(self):
        scores = pd.Series([80, 90])
        returns = pd.Series([-0.01, -0.02])
        hr = hit_rate(scores, returns, threshold=70)
        assert hr == 0.0

    def test_no_signals_above_threshold(self):
        scores = pd.Series([10, 20, 30])
        returns = pd.Series([0.01, 0.02, 0.03])
        hr = hit_rate(scores, returns, threshold=70)
        assert np.isnan(hr)

class TestSortinoRatio:
    def test_all_positive_returns(self):
        returns = pd.Series([0.01, 0.02, 0.01, 0.015, 0.02])
        s = sortino_ratio(returns)
        assert s > 0

    def test_all_negative_returns(self):
        returns = pd.Series([-0.01, -0.02, -0.01, -0.015])
        s = sortino_ratio(returns)
        assert s < 0

    def test_zero_returns(self):
        returns = pd.Series([0.0, 0.0, 0.0, 0.0])
        s = sortino_ratio(returns)
        assert s == 0.0

class TestMaxDrawdown:
    def test_monotonic_increase(self):
        equity = pd.Series([100, 110, 120, 130, 140])
        dd = max_drawdown(equity)
        assert dd == 0.0

    def test_known_drawdown(self):
        equity = pd.Series([100, 110, 90, 95, 85])
        dd = max_drawdown(equity)
        assert dd == pytest.approx(-25 / 110, abs=0.001)

    def test_single_point(self):
        equity = pd.Series([100])
        dd = max_drawdown(equity)
        assert dd == 0.0

class TestCAGR:
    def test_doubling_in_one_year(self):
        equity = pd.Series(np.linspace(100, 200, 253))
        c = cagr(equity, periods_per_year=252)
        assert c == pytest.approx(1.0, abs=0.01)

    def test_flat_equity(self):
        equity = pd.Series([100] * 100)
        c = cagr(equity)
        assert c == 0.0

class TestForwardReturns:
    def test_correct_shift(self):
        prices = pd.Series([100, 110, 105, 120, 115, 130])
        fwd = forward_returns(prices, horizon=2)
        assert fwd.iloc[0] == pytest.approx(0.05, abs=1e-6)
        assert fwd.iloc[-1] != fwd.iloc[-1]
        assert fwd.iloc[-2] != fwd.iloc[-2]

class TestEvaluateSignals:
    def test_no_signals(self):
        entry = pd.Series([10, 20, 30, 40])
        exit_ = pd.Series([10, 20, 30, 40])
        returns = pd.Series([0.01, -0.01, 0.02, -0.02])
        result = evaluate_signals(entry, exit_, returns, 70, 70)
        assert result["n_trades"] == 0
        assert np.isnan(result["avg_holding_period"])

    def test_simple_trade(self):
        n = 20
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        entry = pd.Series([0] * 5 + [80] + [0] * 14, index=idx)
        exit_ = pd.Series([0] * 10 + [80] + [0] * 9, index=idx)
        returns = pd.Series([0.01] * n, index=idx)

        result = evaluate_signals(entry, exit_, returns, 70, 70)
        assert result["n_trades"] >= 1
        assert result["roi_per_trade"] > 0

class TestWalkForwardFolds:
    def test_no_overlap(self):
        folds = _generate_folds(1000, 500, 100, overlap=False)
        assert len(folds) == 5
        for _, _, ts, te in folds:
            assert te - ts == 100

    def test_expanding_train(self):
        folds = _generate_folds(1000, 300, 100, overlap=False, expanding=True)
        for ts, _, _, _ in folds:
            assert ts == 0

    def test_rolling_train(self):
        folds = _generate_folds(1000, 300, 100, overlap=False, expanding=False)
        for ts, te, test_s, _ in folds:
            assert te == test_s
            assert te - ts == 300

    def test_no_test_overlap_between_folds(self):
        folds = _generate_folds(2000, 500, 200, overlap=False)
        for i in range(len(folds) - 1):
            _, _, _, te_i = folds[i]
            _, _, ts_j, _ = folds[i + 1]
            assert ts_j >= te_i

class TestWalkForwardCV:
    def test_runs_without_error(self, synth_df):
        result = walkforward_cv(
            synth_df,
            score_fn=_dummy_score_fn,
            train_window=400,
            test_window=100,
            symbol="SYNTH",
        )
        assert isinstance(result, WalkForwardResult)
        assert result.n_folds > 0
        assert len(result.folds) == result.n_folds

    def test_fold_metrics_populated(self, synth_df):
        result = walkforward_cv(
            synth_df,
            score_fn=_dummy_score_fn,
            train_window=400,
            test_window=100,
            symbol="SYNTH",
        )
        for fold in result.folds:
            if not fold.warnings:
                assert "entry_metrics" in fold.metrics
                assert "signal_metrics" in fold.metrics

    def test_summary_populated(self, synth_df):
        result = walkforward_cv(
            synth_df,
            score_fn=_dummy_score_fn,
            train_window=400,
            test_window=100,
            symbol="SYNTH",
        )
        assert "mean_sortino" in result.summary or len(result.folds) == 0

    def test_serialisation(self, synth_df):
        result = walkforward_cv(
            synth_df,
            score_fn=_dummy_score_fn,
            train_window=400,
            test_window=100,
            symbol="SYNTH",
        )
        d = result.to_dict()
        assert d["symbol"] == "SYNTH"
        assert "folds" in d
        assert "summary" in d

    def test_determinism(self, synth_df):
        r1 = walkforward_cv(
            synth_df, score_fn=_dummy_score_fn,
            train_window=400, test_window=100, symbol="SYNTH",
        )
        r2 = walkforward_cv(
            synth_df, score_fn=_dummy_score_fn,
            train_window=400, test_window=100, symbol="SYNTH",
        )
        assert r1.to_dict() == r2.to_dict()

class TestPurgedKFoldSplits:
    def test_no_leak(self):
        splits = _purged_kfold_splits(1000, 5, embargo=20)
        for train_idx, test_idx in splits:
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0

    def test_embargo_respected(self):
        splits = _purged_kfold_splits(1000, 5, embargo=20)
        for train_idx, test_idx in splits:
            test_min = test_idx.min()
            test_max = test_idx.max()
            near_test = train_idx[
                (train_idx >= test_min - 20) & (train_idx <= test_max + 20)
            ]
            assert len(near_test) == 0

    def test_correct_number_of_splits(self):
        splits = _purged_kfold_splits(1000, 5, embargo=10)
        assert len(splits) == 5

    def test_full_coverage(self):
        splits = _purged_kfold_splits(100, 5, embargo=2)
        all_test = np.concatenate([t for _, t in splits])
        assert len(np.unique(all_test)) == 100

class TestPurgedKFold:
    def test_runs_without_error(self, synth_df):
        result = purged_kfold(
            synth_df,
            score_fn=_dummy_score_fn,
            n_splits=3,
            embargo=10,
            symbol="SYNTH",
        )
        assert isinstance(result, KFoldResult)
        assert len(result.folds) == 3

    def test_metrics_populated(self, synth_df):
        result = purged_kfold(
            synth_df,
            score_fn=_dummy_score_fn,
            n_splits=3,
            embargo=10,
            symbol="SYNTH",
        )
        for fold in result.folds:
            if not fold.warnings:
                assert "entry_metrics" in fold.metrics

    def test_serialisation(self, synth_df):
        result = purged_kfold(
            synth_df,
            score_fn=_dummy_score_fn,
            n_splits=3,
            embargo=10,
            symbol="SYNTH",
        )
        d = result.to_dict()
        assert d["n_splits"] == 3
        assert d["embargo"] == 10

class TestComputeAllMetrics:
    def test_returns_all_keys(self, synth_df, synth_scores):
        entry, exit_ = synth_scores
        prices = synth_df["close"]

        result = compute_all_metrics(entry, exit_, prices)
        assert "entry_metrics" in result
        assert "exit_metrics" in result
        assert "signal_metrics" in result

    def test_entry_metrics_keys(self, synth_df, synth_scores):
        entry, exit_ = synth_scores
        prices = synth_df["close"]

        result = compute_all_metrics(entry, exit_, prices)
        em = result["entry_metrics"]
        assert "ic_5d" in em
        assert "hit_rate_5d" in em
        assert "sortino" in em
        assert "max_drawdown" in em
        assert "cagr" in em

    def test_signal_metrics_keys(self, synth_df, synth_scores):
        entry, exit_ = synth_scores
        prices = synth_df["close"]

        result = compute_all_metrics(entry, exit_, prices)
        sm = result["signal_metrics"]
        assert "n_trades" in sm
        assert "avg_holding_period" in sm
        assert "profit_factor" in sm
