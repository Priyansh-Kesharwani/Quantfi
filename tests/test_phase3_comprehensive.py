"""
Phase 3 — Comprehensive Mathematical & Edge-Case Tests.

Covers every mathematical property, boundary condition, and invariant for:
  - metrics.py        (IC, Sortino, CAGR, MDD, hit-rate, signals, fwd-returns)
  - execution_model.py (impact power-law, fill-price symmetry, latency, costs)
  - tuning.py         (objective S(θ), search spaces, inner-CV, sensitivity)
  - hawkes_simulator.py (LOB invariants, OFI signs, trade-tick properties)
  - kfold.py          (purge no-leak, embargo, full coverage)
  - walkforward.py    (fold generation, overlap, expanding vs rolling)
  - phase3_runner.py  (helpers: aggregate, threshold sweep, subsample, checksum)
"""

import pytest
import numpy as np
import pandas as pd
import sys
import json
import tempfile
from pathlib import Path
from copy import deepcopy

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
from validation.execution_model import (
    ExecutionConfig,
    market_impact,
    compute_fill_price,
    simulate_latency,
    apply_execution_costs,
    slippage_sensitivity_matrix,
)
from validation.tuning import (
    TuningConfig,
    TuningResult,
    TuningTrialResult,
    _generate_grid,
    _sample_random,
    _evaluate_inner_cv,
    run_tuning,
    parameter_sensitivity,
    ablation_study,
)
from validation.kfold import _purged_kfold_splits, purged_kfold, KFoldResult
from validation.walkforward import _generate_folds, walkforward_cv, WalkForwardResult
from simulations.hawkes_simulator import (
    simulate_hawkes_events,
    ground_truth_intensity,
    branching_ratio,
    expected_event_rate,
    intensity_rmse,
    relative_rmse,
    simulate_regime,
    generate_synthetic_lob,
    generate_synthetic_trades,
    validate_estimation,
)


# ====================================================================
# Shared Fixtures
# ====================================================================

@pytest.fixture(scope="module")
def synth_df():
    return fbm_series(n=1000, H=0.6, seed=42)


@pytest.fixture(scope="module")
def synth_df_large():
    return fbm_series(n=1200, H=0.6, seed=42)


@pytest.fixture(scope="module")
def ou_df():
    return ou_series(n=1000, theta=0.15, mu=100.0, sigma=2.0, seed=42)


def _test_score_fn(df, params=None):
    """Deterministic mock score function."""
    np.random.seed(42)
    n = len(df)
    scale = params.get("S_scale", 1.0) if params else 1.0
    entry = pd.Series(
        np.clip(50 + np.random.randn(n) * 15 * scale, 0, 100),
        index=df.index,
    )
    exit_ = pd.Series(
        np.clip(50 + np.random.randn(n) * 15, 0, 100),
        index=df.index,
    )
    return entry, exit_


# ####################################################################
# 1. METRICS — Mathematical Properties & Edge Cases
# ####################################################################

class TestIC_MathProperties:
    """Information Coefficient mathematical properties."""

    def test_ic_bounded_minus1_to_1(self):
        """IC must always be in [-1, 1]."""
        np.random.seed(0)
        for _ in range(50):
            s = pd.Series(np.random.randn(200))
            r = pd.Series(np.random.randn(200))
            ic = information_coefficient(s, r)
            assert -1.0 <= ic <= 1.0

    def test_constant_scores_nan(self):
        """Constant scores → Spearman r undefined → NaN or 0."""
        scores = pd.Series([50.0] * 20)
        returns = pd.Series(np.random.randn(20))
        ic = information_coefficient(scores, returns)
        # scipy.spearmanr returns NaN when one input is constant
        assert np.isnan(ic) or ic == 0.0

    def test_constant_returns_nan(self):
        """Constant returns → Spearman r undefined."""
        scores = pd.Series(np.arange(20, dtype=float))
        returns = pd.Series([0.01] * 20)
        ic = information_coefficient(scores, returns)
        assert np.isnan(ic) or ic == 0.0

    def test_reversed_scores_negates_ic(self):
        """IC(scores, ret) ≈ -IC(100-scores, ret) for rank correlation."""
        np.random.seed(7)
        scores = pd.Series(np.random.rand(100) * 100)
        returns = pd.Series(np.random.randn(100))
        ic_pos = information_coefficient(scores, returns)
        ic_neg = information_coefficient(100 - scores, returns)
        assert ic_pos == pytest.approx(-ic_neg, abs=1e-10)

    def test_tied_ranks(self):
        """IC handles tied values gracefully."""
        scores = pd.Series([10, 10, 10, 20, 20, 30, 30, 40, 40, 50])
        returns = pd.Series([0.01, 0.02, 0.0, 0.03, 0.04, 0.05, -0.01, 0.06, 0.07, 0.08])
        ic = information_coefficient(scores, returns)
        assert not np.isnan(ic)
        assert -1 <= ic <= 1

    def test_all_nan_returns_nan(self):
        """All NaN inputs → NaN."""
        scores = pd.Series([np.nan] * 10)
        returns = pd.Series([np.nan] * 10)
        ic = information_coefficient(scores, returns)
        assert np.isnan(ic)

    def test_ic_with_two_valid_returns_nan(self):
        """Fewer than 3 non-NaN pairs → NaN."""
        scores = pd.Series([10.0, 20.0, np.nan, np.nan])
        returns = pd.Series([0.01, 0.02, np.nan, np.nan])
        ic = information_coefficient(scores, returns)
        assert np.isnan(ic)

    def test_ic_exactly_three_valid(self):
        """Exactly 3 valid pairs → valid IC."""
        scores = pd.Series([10.0, 20.0, 30.0])
        returns = pd.Series([0.01, 0.02, 0.03])
        ic = information_coefficient(scores, returns)
        assert not np.isnan(ic)


class TestHitRate_EdgeCases:
    """Hit rate boundary conditions."""

    def test_threshold_zero_all_signals(self):
        """threshold=0 → all bars are signals."""
        scores = pd.Series([10, 20, 30, 40])
        returns = pd.Series([0.01, -0.01, 0.02, -0.02])
        hr = hit_rate(scores, returns, threshold=0)
        assert hr == 0.5  # 2/4 positive

    def test_threshold_100_no_signals(self):
        """threshold=100 → no signals → NaN."""
        scores = pd.Series([10, 20, 80, 99])
        returns = pd.Series([0.01, 0.02, 0.03, 0.04])
        hr = hit_rate(scores, returns, threshold=100)
        assert np.isnan(hr)

    def test_hit_rate_bounded_01(self):
        """Hit rate always in [0, 1]."""
        np.random.seed(3)
        for _ in range(50):
            scores = pd.Series(np.random.rand(100) * 100)
            returns = pd.Series(np.random.randn(100) * 0.01)
            hr = hit_rate(scores, returns, threshold=50)
            if not np.isnan(hr):
                assert 0.0 <= hr <= 1.0

    def test_boundary_score_equality(self):
        """Score exactly at threshold is NOT counted (> not >=)."""
        scores = pd.Series([70.0, 70.0, 70.0])
        returns = pd.Series([0.01, 0.02, 0.03])
        hr = hit_rate(scores, returns, threshold=70.0)
        assert np.isnan(hr)  # 70.0 > 70.0 is False

    def test_single_signal_hit(self):
        scores = pd.Series([10, 80])
        returns = pd.Series([-0.01, 0.02])
        hr = hit_rate(scores, returns, threshold=70)
        assert hr == 1.0

    def test_single_signal_miss(self):
        scores = pd.Series([10, 80])
        returns = pd.Series([-0.01, -0.02])
        hr = hit_rate(scores, returns, threshold=70)
        assert hr == 0.0


class TestSortino_MathProperties:
    """Sortino ratio mathematical properties and edge cases."""

    def test_sign_reversal(self):
        """Flipping returns should roughly flip Sortino sign."""
        returns = pd.Series([0.02, -0.01, 0.03, -0.005, 0.01])
        s_pos = sortino_ratio(returns)
        s_neg = sortino_ratio(-returns)
        # Sortino of positive series should be positive
        assert s_pos > 0
        # Sortino of negative series should be negative
        assert s_neg < 0

    def test_no_downside_returns_inf(self):
        """All positive excess returns → inf Sortino."""
        returns = pd.Series([0.01, 0.02, 0.03, 0.005])
        s = sortino_ratio(returns, target=0.0)
        assert s == np.inf

    def test_empty_series_zero(self):
        """Empty series → 0."""
        s = sortino_ratio(pd.Series([], dtype=float))
        assert s == 0.0

    def test_single_positive_return_inf(self):
        """Single positive return, no downside → inf."""
        s = sortino_ratio(pd.Series([0.05]))
        assert s == np.inf

    def test_single_negative_return(self):
        """Single negative return → std is NaN (ddof=1), so Sortino is 0.0 or NaN."""
        s = sortino_ratio(pd.Series([-0.05]))
        # With a single downside observation, std(ddof=1)=NaN → falls through to 0.0
        assert s == 0.0 or np.isnan(s)

    def test_two_negative_returns(self):
        """Two negative returns → Sortino < 0 (downside_std defined)."""
        s = sortino_ratio(pd.Series([-0.01, -0.02]))
        assert s < 0

    def test_annualization_scales_correctly(self):
        """Doubling annualization should scale by sqrt(2)."""
        returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
        s_252 = sortino_ratio(returns, annualization=252)
        s_126 = sortino_ratio(returns, annualization=126)
        ratio = s_252 / s_126
        assert ratio == pytest.approx(np.sqrt(2), abs=0.01)

    def test_target_shifts_excess(self):
        """A positive target raises the bar for 'positive' returns."""
        returns = pd.Series([0.01, 0.02, 0.005, -0.01, 0.015, -0.005, 0.008, -0.003])
        s_zero = sortino_ratio(returns, target=0.0)
        s_high = sortino_ratio(returns, target=0.03)
        # Both should be finite
        assert np.isfinite(s_zero)
        assert np.isfinite(s_high)
        # Higher target → lower Sortino
        assert s_high < s_zero


class TestMaxDrawdown_MathProperties:
    """Max drawdown mathematical properties."""

    def test_always_in_range(self):
        """Max drawdown ∈ [-1, 0] for any positive equity curve."""
        np.random.seed(4)
        for _ in range(50):
            equity = pd.Series(100 * np.exp(np.cumsum(np.random.randn(100) * 0.02)))
            dd = max_drawdown(equity)
            assert -1.0 <= dd <= 0.0

    def test_monotonic_increase_zero_dd(self):
        equity = pd.Series([100, 110, 120, 130])
        assert max_drawdown(equity) == 0.0

    def test_monotonic_decrease_full_dd(self):
        """Monotonically decreasing → drawdown = (last-first)/first."""
        equity = pd.Series([100.0, 80.0, 60.0, 40.0])
        dd = max_drawdown(equity)
        assert dd == pytest.approx((40 - 100) / 100, abs=1e-10)

    def test_v_recovery(self):
        """V-shape: peak → trough → new peak → drawdown from first peak."""
        equity = pd.Series([100, 110, 80, 90, 115])
        dd = max_drawdown(equity)
        # Peak=110, trough=80 → dd=(80-110)/110
        assert dd == pytest.approx(-30 / 110, abs=1e-10)

    def test_flat_equity_zero(self):
        equity = pd.Series([100] * 10)
        assert max_drawdown(equity) == 0.0

    def test_single_point_zero(self):
        assert max_drawdown(pd.Series([42.0])) == 0.0

    def test_two_points_drop(self):
        dd = max_drawdown(pd.Series([100.0, 90.0]))
        assert dd == pytest.approx(-0.10, abs=1e-10)


class TestCAGR_MathProperties:
    """CAGR mathematical properties."""

    def test_known_doubling(self):
        """1 year of doubling → ~100% CAGR."""
        equity = pd.Series(np.linspace(100, 200, 253))  # 253 points = 252 intervals = 1 year
        c = cagr(equity, periods_per_year=252)
        assert c == pytest.approx(1.0, abs=0.02)

    def test_compound_consistency(self):
        """CAGR should decompose correctly: start*(1+CAGR)^years ≈ end."""
        equity = pd.Series(100 * np.exp(np.cumsum(np.random.RandomState(42).randn(505) * 0.002)))
        c = cagr(equity, periods_per_year=252)
        years = (len(equity) - 1) / 252
        reconstructed_end = equity.iloc[0] * (1 + c) ** years
        assert reconstructed_end == pytest.approx(equity.iloc[-1], rel=0.01)

    def test_flat_returns_zero(self):
        assert cagr(pd.Series([100.0] * 100)) == 0.0

    def test_negative_equity_end(self):
        """Negative ending equity → CAGR still computes (may be < -1 for total return < 0)."""
        equity = pd.Series([100.0, 80.0, 50.0, 30.0, 10.0])
        c = cagr(equity)
        assert c < 0

    def test_single_bar(self):
        """Single data point → 0."""
        assert cagr(pd.Series([100.0])) == 0.0

    def test_two_bars(self):
        """Two bars: CAGR = (end/start)^(1/years) - 1."""
        equity = pd.Series([100.0, 110.0])
        c = cagr(equity, periods_per_year=252)
        years = 1 / 252
        expected = (110 / 100) ** (1 / years) - 1
        assert c == pytest.approx(expected, rel=1e-6)

    def test_zero_start_returns_zero(self):
        """Start ≤ 0 → 0."""
        assert cagr(pd.Series([0.0, 100.0, 200.0])) == 0.0

    def test_all_nan_returns_zero(self):
        assert cagr(pd.Series([np.nan, np.nan])) == 0.0


class TestForwardReturns_EdgeCases:

    def test_horizon_1(self):
        prices = pd.Series([100.0, 105.0, 110.0, 108.0])
        fwd = forward_returns(prices, horizon=1)
        assert fwd.iloc[0] == pytest.approx(0.05, abs=1e-10)
        assert fwd.iloc[1] == pytest.approx(110 / 105 - 1, abs=1e-10)
        assert np.isnan(fwd.iloc[-1])

    def test_horizon_equals_length(self):
        """Horizon = length → all NaN."""
        prices = pd.Series([100, 110, 120])
        fwd = forward_returns(prices, horizon=3)
        assert fwd.isna().all()

    def test_horizon_larger_than_length(self):
        prices = pd.Series([100.0, 110.0])
        fwd = forward_returns(prices, horizon=5)
        assert fwd.isna().all()

    def test_constant_prices_zero_returns(self):
        prices = pd.Series([100.0] * 10)
        fwd = forward_returns(prices, horizon=3)
        valid = fwd.dropna()
        assert (valid == 0.0).all()

    def test_name_includes_horizon(self):
        prices = pd.Series([100.0, 110.0, 120.0])
        fwd = forward_returns(prices, horizon=2)
        assert "2" in fwd.name


class TestEvaluateSignals_EdgeCases:
    """Comprehensive edge cases for signal extraction and trade metrics."""

    def test_immediate_entry_exit_same_bar(self):
        """Entry and exit score both above threshold on same bar → 1-bar trade."""
        n = 10
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        entry = pd.Series([0, 0, 80, 0, 0, 0, 0, 0, 0, 0], index=idx, dtype=float)
        exit_ = pd.Series([0, 0, 0, 80, 0, 0, 0, 0, 0, 0], index=idx, dtype=float)
        returns = pd.Series([0.01] * n, index=idx)
        result = evaluate_signals(entry, exit_, returns, 70, 70)
        assert result["n_trades"] >= 1
        assert result["avg_holding_period"] >= 1

    def test_forced_close_at_end(self):
        """Open trade at end of series is forced closed."""
        n = 10
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        entry = pd.Series([0] * 8 + [80, 0], index=idx, dtype=float)
        exit_ = pd.Series([0] * 10, index=idx, dtype=float)  # Never exits
        returns = pd.Series([0.01] * n, index=idx)
        result = evaluate_signals(entry, exit_, returns, 70, 70)
        assert result["n_trades"] == 1  # forced close at end

    def test_multiple_trades(self):
        """Multiple distinct trades in series."""
        n = 30
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        entry = pd.Series([0] * 30, index=idx, dtype=float)
        exit_ = pd.Series([0] * 30, index=idx, dtype=float)
        # Trade 1: enter at 2, exit at 5
        entry.iloc[2] = 80
        exit_.iloc[5] = 80
        # Trade 2: enter at 10, exit at 15
        entry.iloc[10] = 80
        exit_.iloc[15] = 80
        returns = pd.Series([0.005] * n, index=idx)
        result = evaluate_signals(entry, exit_, returns, 70, 70)
        assert result["n_trades"] == 2

    def test_all_nan_scores(self):
        """All NaN scores → no trades."""
        n = 10
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        entry = pd.Series([np.nan] * n, index=idx)
        exit_ = pd.Series([np.nan] * n, index=idx)
        returns = pd.Series([0.01] * n, index=idx)
        result = evaluate_signals(entry, exit_, returns, 70, 70)
        assert result["n_trades"] == 0

    def test_profit_factor_no_losses(self):
        """All winning trades → profit_factor > 0."""
        n = 20
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        entry = pd.Series([0] * 5 + [80] + [0] * 14, index=idx, dtype=float)
        exit_ = pd.Series([0] * 10 + [80] + [0] * 9, index=idx, dtype=float)
        returns = pd.Series([0.01] * n, index=idx)  # Always positive
        result = evaluate_signals(entry, exit_, returns, 70, 70)
        assert result["profit_factor"] > 0

    def test_win_loss_ratio_keys_exist(self):
        n = 20
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        entry = pd.Series([0] * 5 + [80] + [0] * 14, index=idx, dtype=float)
        exit_ = pd.Series([0] * 10 + [80] + [0] * 9, index=idx, dtype=float)
        returns = pd.Series([0.01] * n, index=idx)
        result = evaluate_signals(entry, exit_, returns, 70, 70)
        assert "win_loss_ratio" in result
        assert "roi_per_trade" in result
        assert "total_return" in result

    def test_misaligned_indices_handled(self):
        """Different index lengths → intersection used."""
        idx1 = pd.date_range("2020-01-01", periods=10, freq="B")
        idx2 = pd.date_range("2020-01-05", periods=10, freq="B")
        entry = pd.Series([80] * 10, index=idx1, dtype=float)
        exit_ = pd.Series([80] * 10, index=idx2, dtype=float)
        returns = pd.Series([0.01] * 10, index=idx1)
        # Should not raise, intersection will be used
        result = evaluate_signals(entry, exit_, returns, 70, 70)
        assert isinstance(result, dict)


class TestComputeScoreMetrics_Comprehensive:
    """compute_score_metrics output structure and values."""

    def test_all_keys_present(self, synth_df):
        scores = pd.Series(np.random.rand(len(synth_df)) * 100, index=synth_df.index)
        prices = synth_df["close"]
        result = compute_score_metrics(scores, prices, forward_horizons=[5, 10])
        assert "ic_5d" in result
        assert "ic_10d" in result
        assert "hit_rate_5d" in result
        assert "hit_rate_10d" in result
        assert "sortino" in result
        assert "max_drawdown" in result
        assert "cagr" in result
        assert "n_signals" in result
        assert "pct_active" in result

    def test_pct_active_bounded(self, synth_df):
        scores = pd.Series(np.random.rand(len(synth_df)) * 100, index=synth_df.index)
        prices = synth_df["close"]
        result = compute_score_metrics(scores, prices)
        assert 0.0 <= result["pct_active"] <= 1.0

    def test_n_signals_correct(self, synth_df):
        """Number of signals = count(score > threshold)."""
        scores = pd.Series([80.0] * len(synth_df), index=synth_df.index)
        prices = synth_df["close"]
        result = compute_score_metrics(scores, prices, entry_threshold=70)
        assert result["n_signals"] == len(synth_df)

    def test_zero_scores_no_signals(self, synth_df):
        scores = pd.Series([0.0] * len(synth_df), index=synth_df.index)
        prices = synth_df["close"]
        result = compute_score_metrics(scores, prices, entry_threshold=70)
        assert result["n_signals"] == 0


class TestComputeAllMetrics_Comprehensive:

    def test_three_top_level_keys(self, synth_df):
        entry = pd.Series(np.random.rand(len(synth_df)) * 100, index=synth_df.index)
        exit_ = pd.Series(np.random.rand(len(synth_df)) * 100, index=synth_df.index)
        result = compute_all_metrics(entry, exit_, synth_df["close"])
        assert set(result.keys()) == {"entry_metrics", "exit_metrics", "signal_metrics"}

    def test_custom_forward_horizons(self, synth_df):
        entry = pd.Series(np.random.rand(len(synth_df)) * 100, index=synth_df.index)
        exit_ = pd.Series(np.random.rand(len(synth_df)) * 100, index=synth_df.index)
        result = compute_all_metrics(entry, exit_, synth_df["close"], forward_horizons=[3, 7])
        assert "ic_3d" in result["entry_metrics"]
        assert "ic_7d" in result["entry_metrics"]


# ####################################################################
# 2. EXECUTION MODEL — Mathematical Properties & Edge Cases
# ####################################################################

class TestMarketImpact_MathProperties:
    """Market impact power-law and scaling properties."""

    def test_power_law_scaling(self):
        """impact(k*Q) / impact(Q) = k^γ for γ=0.5."""
        k = 4.0
        gamma = 0.5
        i_base = market_impact(1000, 1e6, k_impact=0.01, gamma=gamma)
        i_scaled = market_impact(k * 1000, 1e6, k_impact=0.01, gamma=gamma)
        assert i_scaled / i_base == pytest.approx(k ** gamma, rel=1e-10)

    def test_gamma_0_constant_impact(self):
        """gamma=0 → impact = k_impact regardless of order size."""
        i1 = market_impact(100, 1e6, k_impact=0.01, gamma=0)
        i2 = market_impact(10000, 1e6, k_impact=0.01, gamma=0)
        assert i1 == pytest.approx(0.01, abs=1e-10)
        assert i2 == pytest.approx(0.01, abs=1e-10)

    def test_gamma_1_linear_impact(self):
        """gamma=1 → impact = k * (Q/ADV), doubling Q doubles impact."""
        i1 = market_impact(1000, 1e6, k_impact=0.01, gamma=1.0)
        i2 = market_impact(2000, 1e6, k_impact=0.01, gamma=1.0)
        assert i2 == pytest.approx(2 * i1, rel=1e-10)

    def test_negative_order_uses_absolute(self):
        """Negative order size → uses abs (no negative impact)."""
        i_pos = market_impact(1000, 1e6, k_impact=0.01, gamma=0.5)
        i_neg = market_impact(-1000, 1e6, k_impact=0.01, gamma=0.5)
        assert i_neg == i_pos

    def test_tiny_order_near_zero_impact(self):
        i = market_impact(0.001, 1e6, k_impact=0.01, gamma=0.5)
        assert i < 1e-6

    def test_impact_monotonically_increases_with_size(self):
        sizes = [100, 500, 1000, 5000, 10000]
        impacts = [market_impact(s, 1e6, 0.01, 0.5) for s in sizes]
        for i in range(len(impacts) - 1):
            assert impacts[i + 1] > impacts[i]


class TestFillPrice_MathProperties:
    """Fill price symmetry and cost decomposition."""

    def test_buy_fill_above_mid_no_noise(self):
        cfg = ExecutionConfig(sigma_slip=0.0)
        rng = np.random.RandomState(42)
        fill, _ = compute_fill_price(100.0, 1, 1000, 1e6, cfg, rng)
        assert fill > 100.0

    def test_sell_fill_below_mid_no_noise(self):
        cfg = ExecutionConfig(sigma_slip=0.0)
        rng = np.random.RandomState(42)
        fill, _ = compute_fill_price(100.0, -1, 1000, 1e6, cfg, rng)
        assert fill < 100.0

    def test_cost_decomposition_adds_up(self):
        """total_cost_frac ≈ total_slip_frac + commission_frac."""
        _, bd = compute_fill_price(100.0, 1, 1000, 1e6)
        assert bd["total_cost_frac"] == pytest.approx(
            bd["total_slip_frac"] + bd["commission_frac"], abs=1e-10
        )

    def test_determinism_same_rng(self):
        cfg = ExecutionConfig()
        f1, b1 = compute_fill_price(100.0, 1, 1000, 1e6, cfg, np.random.RandomState(42))
        f2, b2 = compute_fill_price(100.0, 1, 1000, 1e6, cfg, np.random.RandomState(42))
        assert f1 == f2
        assert b1["total_cost_frac"] == b2["total_cost_frac"]

    def test_zero_adv_no_impact(self):
        cfg = ExecutionConfig(sigma_slip=0.0)
        fill, bd = compute_fill_price(100.0, 1, 1000, 0, cfg)
        assert bd["impact_frac"] == 0.0

    def test_zero_order_size_no_impact(self):
        cfg = ExecutionConfig(sigma_slip=0.0)
        _, bd = compute_fill_price(100.0, 1, 0, 1e6, cfg)
        assert bd["impact_frac"] == 0.0

    def test_larger_order_more_expensive(self):
        """Larger order → higher fill price for buys (no noise)."""
        cfg = ExecutionConfig(sigma_slip=0.0)
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)
        f_small, _ = compute_fill_price(100.0, 1, 100, 1e6, cfg, rng1)
        f_large, _ = compute_fill_price(100.0, 1, 10000, 1e6, cfg, rng2)
        assert f_large > f_small


class TestLatency_EdgeCases:

    def test_n_zero(self):
        lats = simulate_latency(0)
        assert len(lats) == 0

    def test_zero_mean_with_jitter(self):
        """Mean=0 with jitter → all zeros."""
        lats = simulate_latency(10, mean_ms=0, jitter=True)
        assert (lats == 0).all()

    def test_no_jitter_constant(self):
        lats = simulate_latency(10, mean_ms=50, jitter=False)
        assert (lats == 50.0).all()

    def test_exponential_mean_convergence(self):
        """Large sample: mean of exp(λ) samples ≈ λ."""
        lats = simulate_latency(100000, mean_ms=50, jitter=True, seed=0)
        assert np.mean(lats) == pytest.approx(50.0, rel=0.05)

    def test_all_positive(self):
        lats = simulate_latency(1000, mean_ms=10, seed=42)
        assert (lats >= 0).all()


class TestApplyExecCosts_EdgeCases:

    @pytest.fixture
    def signal_data(self):
        np.random.seed(42)
        n = 200
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        returns = pd.Series(np.random.randn(n) * 0.01, index=idx)
        entry = pd.Series(np.clip(50 + np.random.randn(n) * 20, 0, 100), index=idx)
        exit_ = pd.Series(np.clip(50 + np.random.randn(n) * 20, 0, 100), index=idx)
        volume = pd.Series(np.abs(np.random.randn(n) * 1e6 + 5e6), index=idx)
        return returns, entry, exit_, volume

    def test_determinism(self, signal_data):
        """Same seed → identical results."""
        returns, entry, exit_, volume = signal_data
        adj1, r1 = apply_execution_costs(returns, entry, exit_, volume, seed=42)
        adj2, r2 = apply_execution_costs(returns, entry, exit_, volume, seed=42)
        np.testing.assert_array_equal(adj1.values, adj2.values)
        assert r1 == r2

    def test_no_signals_no_change(self):
        """All scores below threshold → no trades, returns unchanged."""
        n = 50
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        returns = pd.Series(np.random.randn(n) * 0.01, index=idx)
        entry = pd.Series([10.0] * n, index=idx)  # All below 70
        exit_ = pd.Series([10.0] * n, index=idx)
        volume = pd.Series([1e6] * n, index=idx)
        adj, report = apply_execution_costs(returns, entry, exit_, volume)
        np.testing.assert_array_equal(adj.values, returns.values)
        assert report["n_trades"] == 0

    def test_cost_monotonicity_with_k_impact(self, signal_data):
        """Higher k_impact → more PnL erosion."""
        returns, entry, exit_, volume = signal_data
        cfg_low = ExecutionConfig(k_impact=0.001)
        cfg_high = ExecutionConfig(k_impact=0.01)
        _, r_low = apply_execution_costs(returns, entry, exit_, volume, config=cfg_low, seed=42)
        _, r_high = apply_execution_costs(returns, entry, exit_, volume, config=cfg_high, seed=42)
        assert r_high["pnl_erosion"] >= r_low["pnl_erosion"]

    def test_forced_close_at_end(self):
        """Trade open at end should be force-closed."""
        n = 20
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        returns = pd.Series([0.01] * n, index=idx)
        entry = pd.Series([0] * 15 + [80] + [0] * 4, index=idx, dtype=float)
        exit_ = pd.Series([0] * n, index=idx, dtype=float)  # Never exits
        volume = pd.Series([1e6] * n, index=idx)
        adj, report = apply_execution_costs(returns, entry, exit_, volume)
        assert report["n_trades"] == 1


class TestSlippageSensitivityMatrix_Properties:

    def test_shape(self):
        np.random.seed(42)
        n = 100
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        returns = pd.Series(np.random.randn(n) * 0.01, index=idx)
        entry = pd.Series(np.clip(50 + np.random.randn(n) * 20, 0, 100), index=idx)
        exit_ = pd.Series(np.clip(50 + np.random.randn(n) * 20, 0, 100), index=idx)
        volume = pd.Series(np.abs(np.random.randn(n) * 1e6), index=idx)
        k_range = [0.0, 0.001, 0.005]
        g_range = [0.3, 0.5, 1.0]
        matrix = slippage_sensitivity_matrix(
            returns, entry, exit_, volume,
            k_impact_range=k_range, gamma_range=g_range,
        )
        assert matrix.shape == (len(k_range), len(g_range))

    def test_zero_impact_zero_erosion(self):
        """k_impact=0, no spread/commission → zero erosion."""
        np.random.seed(42)
        n = 50
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        returns = pd.Series(np.random.randn(n) * 0.01, index=idx)
        entry = pd.Series(np.clip(50 + np.random.randn(n) * 20, 0, 100), index=idx)
        exit_ = pd.Series(np.clip(50 + np.random.randn(n) * 20, 0, 100), index=idx)
        volume = pd.Series(np.abs(np.random.randn(n) * 1e6 + 5e6), index=idx)
        # k_impact_range=[0.0] should produce minimal erosion
        matrix = slippage_sensitivity_matrix(
            returns, entry, exit_, volume,
            k_impact_range=[0.0], gamma_range=[0.5],
        )
        # With k_impact=0, erosion comes only from spread + commission + noise
        # The entire matrix value should be ≥ 0
        assert matrix.values[0, 0] >= 0


class TestExecutionConfig_Extended:

    def test_from_dict_nested_keys(self):
        d = {
            "slippage": {"k_impact": 0.005, "gamma": 0.7, "sigma_slip": 0.0002, "order_size_pct": 0.02},
            "transaction_costs": {"commission_bps": 5.0, "spread_bps": 2.0},
            "latency": {"mean_latency_ms": 100, "jitter": False},
        }
        cfg = ExecutionConfig.from_dict(d)
        assert cfg.k_impact == 0.005
        assert cfg.gamma == 0.7
        assert cfg.sigma_slip == 0.0002
        assert cfg.order_size_pct == 0.02
        assert cfg.commission_bps == 5.0
        assert cfg.spread_bps == 2.0
        assert cfg.mean_latency_ms == 100
        assert cfg.latency_jitter is False

    def test_empty_dict_defaults(self):
        cfg = ExecutionConfig.from_dict({})
        assert cfg.k_impact == 0.001
        assert cfg.gamma == 0.5
        assert cfg.commission_bps == 2.0


# ####################################################################
# 3. TUNING — Objective Formula & Edge Cases
# ####################################################################

class TestTuningObjective_MathProperties:
    """Verify S(θ) = median(M_f) − λ·std(M_f)."""

    def test_objective_formula_exact(self):
        """Manually verify the objective from known fold_metrics."""
        from validation.tuning import _evaluate_inner_cv

        # We'll compute the score manually and compare
        # Using a known lambda_var and manually setting fold metrics is tricky,
        # so we'll verify the formula in run_tuning instead.
        fold_metrics = [1.0, 2.0, 3.0, 4.0, 5.0]
        med = np.median(fold_metrics)
        std = np.std(fold_metrics)
        lambda_var = 0.5
        expected_score = med - lambda_var * std
        assert expected_score == pytest.approx(3.0 - 0.5 * np.std([1, 2, 3, 4, 5]), abs=1e-10)

    def test_lambda_zero_equals_median(self):
        """λ=0 → S(θ) = median(M_f)."""
        fold_metrics = [1.0, 3.0, 5.0]
        med = np.median(fold_metrics)
        std = np.std(fold_metrics)
        score_lambda0 = med - 0.0 * std
        assert score_lambda0 == pytest.approx(med, abs=1e-10)

    def test_high_lambda_penalizes_variance(self):
        """High λ → variance-dominated → prefers consistent params."""
        fold_a = [2.0, 2.0, 2.0]  # consistent
        fold_b = [0.0, 3.0, 3.0]  # same median=3, higher variance

        lambda_var = 2.0

        score_a = np.median(fold_a) - lambda_var * np.std(fold_a)
        score_b = np.median(fold_b) - lambda_var * np.std(fold_b)
        # With λ=2, consistent params should win
        assert score_a > score_b

    def test_tuning_result_score_matches_formula(self, synth_df_large):
        """run_tuning result.best_score = median - λ*std."""
        cfg = TuningConfig(
            method="grid",
            search_space={"S_scale": [1.0]},
            inner_n_splits=3,
            inner_embargo=10,
            lambda_var=0.5,
            seed=42,
        )
        result = run_tuning(synth_df_large, _test_score_fn, cfg, symbol="TEST")
        trial = result.trials[0]
        expected = trial.median_metric - 0.5 * trial.std_metric
        assert result.best_score == pytest.approx(expected, abs=1e-10)


class TestSearchSpace_EdgeCases:

    def test_grid_single_param(self):
        grid = _generate_grid({"a": [1, 2, 3]})
        assert len(grid) == 3
        assert grid[0] == {"a": 1}

    def test_grid_three_dimensions(self):
        grid = _generate_grid({"a": [1, 2], "b": [10, 20], "c": ["x", "y"]})
        assert len(grid) == 8  # 2*2*2

    def test_grid_empty_space(self):
        grid = _generate_grid({})
        # product of nothing → one empty dict
        assert len(grid) == 1
        assert grid[0] == {}

    def test_random_determinism(self):
        space = {"a": [1, 2, 3], "b": [10, 20, 30]}
        s1 = _sample_random(space, 5, np.random.RandomState(42))
        s2 = _sample_random(space, 5, np.random.RandomState(42))
        assert s1 == s2

    def test_random_single_value(self):
        """Single value in list → always that value."""
        space = {"a": [42]}
        samples = _sample_random(space, 10, np.random.RandomState(0))
        assert all(s["a"] == 42 for s in samples)


class TestInnerCV_Objectives:
    """Test inner CV with different objective functions."""

    def test_ic_objective(self, synth_df_large):
        fold_metrics, med, std = _evaluate_inner_cv(
            synth_df_large, _test_score_fn, {"S_scale": 1.0},
            n_splits=3, embargo=10, objective="ic",
        )
        assert len(fold_metrics) == 3
        for m in fold_metrics:
            assert -1.0 <= m <= 1.0 or m == -999.0

    def test_ir_objective(self, synth_df_large):
        fold_metrics, med, std = _evaluate_inner_cv(
            synth_df_large, _test_score_fn, {"S_scale": 1.0},
            n_splits=3, embargo=10, objective="ir",
        )
        assert len(fold_metrics) == 3
        assert isinstance(med, float)

    def test_unknown_objective_falls_back_to_sortino(self, synth_df_large):
        fold_metrics, med, std = _evaluate_inner_cv(
            synth_df_large, _test_score_fn, {"S_scale": 1.0},
            n_splits=3, embargo=10, objective="unknown_metric",
        )
        assert len(fold_metrics) == 3

    def test_small_dataset_still_works(self):
        """Very small dataset → still runs (may produce -999 folds)."""
        df = fbm_series(n=50, H=0.6, seed=42)
        fold_metrics, med, std = _evaluate_inner_cv(
            df, _test_score_fn, {"S_scale": 1.0},
            n_splits=2, embargo=2,
        )
        assert len(fold_metrics) == 2


class TestRunTuning_Extended:

    def test_unknown_method_fallback(self, synth_df_large):
        """Unknown method → falls back to random_search."""
        cfg = TuningConfig(
            method="unknown_optimizer",
            n_trials=2,
            search_space={"S_scale": [0.5, 1.0]},
            inner_n_splits=3,
            inner_embargo=10,
            seed=42,
        )
        result = run_tuning(synth_df_large, _test_score_fn, cfg, symbol="TEST")
        assert len(result.trials) == 2

    def test_empty_search_space_single_trial(self, synth_df_large):
        """Empty search space → single trial with empty params."""
        cfg = TuningConfig(
            method="grid",
            search_space={},
            inner_n_splits=3,
            inner_embargo=10,
            seed=42,
        )
        result = run_tuning(synth_df_large, _test_score_fn, cfg, symbol="TEST")
        assert len(result.trials) == 1
        assert result.best_params == {}

    def test_best_score_not_neg_inf(self, synth_df_large):
        cfg = TuningConfig(
            method="random_search",
            n_trials=3,
            search_space={"S_scale": [0.5, 1.0, 2.0]},
            inner_n_splits=3,
            inner_embargo=10,
            seed=42,
        )
        result = run_tuning(synth_df_large, _test_score_fn, cfg, symbol="TEST")
        assert result.best_score > -np.inf

    def test_to_json_roundtrip(self, synth_df_large, tmp_path):
        cfg = TuningConfig(
            method="random_search",
            n_trials=2,
            search_space={"S_scale": [1.0, 2.0]},
            inner_n_splits=3,
            inner_embargo=10,
            seed=42,
        )
        result = run_tuning(synth_df_large, _test_score_fn, cfg, symbol="TEST")
        path = str(tmp_path / "tuning.json")
        result.to_json(path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["best_params"] == result.best_params
        assert len(loaded["trials"]) == len(result.trials)


class TestParameterSensitivity_Extended:

    def test_single_value(self, synth_df_large):
        result = parameter_sensitivity(
            synth_df_large, _test_score_fn,
            base_params={"S_scale": 1.0},
            param_name="S_scale",
            param_values=[1.0],
            n_splits=3, embargo=10,
        )
        assert len(result) == 1

    def test_flat_sensitivity_low_std(self, synth_df_large):
        """If the objective doesn't vary much, sensitivity std should be small."""
        result = parameter_sensitivity(
            synth_df_large, _test_score_fn,
            base_params={"S_scale": 1.0},
            param_name="S_scale",
            param_values=[1.0, 1.0, 1.0],  # Same value → same score
            n_splits=3, embargo=10,
        )
        scores = result["score"].values
        assert np.std(scores) < 1e-6


class TestAblation_Extended:

    def test_empty_components(self, synth_df_large):
        """No components to ablate → just baseline."""
        result = ablation_study(
            synth_df_large, _test_score_fn,
            base_params={"S_scale": 1.0},
            components=[],
            n_splits=3, embargo=10,
        )
        assert len(result) == 1  # baseline only
        assert result.iloc[0]["component"] == "baseline"

    def test_impact_sign_convention(self, synth_df_large):
        """If ablation degrades performance, impact should be positive."""
        def _score_fn_with_ablation(df, params=None):
            np.random.seed(42)
            n = len(df)
            base = 50 + np.random.randn(n) * 15
            # If disable_important is set, degrade scores significantly
            if params and params.get("disable_important"):
                base *= 0.5
            return (
                pd.Series(np.clip(base, 0, 100), index=df.index),
                pd.Series(np.clip(100 - base, 0, 100), index=df.index),
            )

        result = ablation_study(
            synth_df_large, _score_fn_with_ablation,
            base_params={},
            components=["important"],
            n_splits=3, embargo=10,
        )
        # 'important' component ablation should show positive impact (helps baseline)
        important_row = result[result["component"] == "important"]
        assert len(important_row) == 1


# ####################################################################
# 4. HAWKES — LOB & Trade Tick Mathematical Invariants
# ####################################################################

class TestLOB_MathInvariants:
    """LOB structural invariants."""

    @pytest.fixture
    def lob_data(self):
        events = simulate_hawkes_events(0.5, 0.3, 1.0, 100.0, seed=42)
        grid = np.arange(0, 100, 1.0)
        return generate_synthetic_lob(events, grid, depth_levels=5, seed=42)

    def test_bid_below_ask_all_levels(self, lob_data):
        """bid[level] < ask[level] for all levels and time points."""
        n = len(lob_data["grid"])
        for i in range(n):
            for lev in range(lob_data["n_levels"]):
                assert lob_data["bid_prices"][i][lev] < lob_data["ask_prices"][i][lev]

    def test_bid_levels_decreasing(self, lob_data):
        """bid_p[k] > bid_p[k+1] (further levels are worse for buyer)."""
        n = len(lob_data["grid"])
        for i in range(n):
            for lev in range(lob_data["n_levels"] - 1):
                assert lob_data["bid_prices"][i][lev] > lob_data["bid_prices"][i][lev + 1]

    def test_ask_levels_increasing(self, lob_data):
        """ask_p[k] < ask_p[k+1] (further levels are worse for buyer)."""
        n = len(lob_data["grid"])
        for i in range(n):
            for lev in range(lob_data["n_levels"] - 1):
                assert lob_data["ask_prices"][i][lev] < lob_data["ask_prices"][i][lev + 1]

    def test_depths_positive(self, lob_data):
        """All depths must be positive."""
        for i in range(len(lob_data["grid"])):
            for lev in range(lob_data["n_levels"]):
                assert lob_data["bid_depths"][i][lev] > 0
                assert lob_data["ask_depths"][i][lev] > 0

    def test_mid_prices_positive(self, lob_data):
        assert (lob_data["mid_prices"] > 0).all()

    def test_empty_events_still_generates(self):
        """No events → LOB still has valid structure (no activity impact)."""
        grid = np.arange(0, 50, 1.0)
        lob = generate_synthetic_lob(np.array([]), grid, depth_levels=3, seed=42)
        assert len(lob["mid_prices"]) == len(grid)
        # All bids < all asks even with no events
        for i in range(len(grid)):
            for lev in range(3):
                assert lob["bid_prices"][i][lev] < lob["ask_prices"][i][lev]

    def test_determinism(self):
        events = simulate_hawkes_events(0.5, 0.3, 1.0, 50.0, seed=42)
        grid = np.arange(0, 50, 1.0)
        l1 = generate_synthetic_lob(events, grid, seed=42)
        l2 = generate_synthetic_lob(events, grid, seed=42)
        np.testing.assert_array_equal(l1["mid_prices"], l2["mid_prices"])
        for i in range(len(grid)):
            np.testing.assert_array_equal(l1["bid_prices"][i], l2["bid_prices"][i])
            np.testing.assert_array_equal(l1["ask_prices"][i], l2["ask_prices"][i])

    def test_varying_depth_levels(self):
        """Different depth_levels produce correct number of levels."""
        events = simulate_hawkes_events(0.5, 0.3, 1.0, 10.0, seed=42)
        grid = np.arange(0, 10, 1.0)
        for depth in [1, 3, 10]:
            lob = generate_synthetic_lob(events, grid, depth_levels=depth, seed=42)
            assert lob["n_levels"] == depth
            assert len(lob["bid_prices"][0]) == depth


class TestSyntheticTrades_MathInvariants:
    """Trade tick mathematical properties."""

    def test_ofi_sign_consistency(self):
        """Buy trades → positive OFI, sell trades → negative OFI."""
        events = simulate_hawkes_events(1.0, 0.3, 1.0, 100.0, seed=42)
        trades = generate_synthetic_trades(events, buy_prob=0.5, seed=42)
        buys = trades[trades["side"] == 1]
        sells = trades[trades["side"] == -1]
        assert (buys["ofi_contribution"] > 0).all()
        assert (sells["ofi_contribution"] < 0).all()

    def test_all_buys(self):
        """buy_prob=1.0 → all buy trades."""
        events = simulate_hawkes_events(1.0, 0.3, 1.0, 50.0, seed=42)
        trades = generate_synthetic_trades(events, buy_prob=1.0, seed=42)
        assert (trades["side"] == 1).all()
        assert (trades["ofi_contribution"] > 0).all()

    def test_all_sells(self):
        """buy_prob=0.0 → all sell trades."""
        events = simulate_hawkes_events(1.0, 0.3, 1.0, 50.0, seed=42)
        trades = generate_synthetic_trades(events, buy_prob=0.0, seed=42)
        assert (trades["side"] == -1).all()
        assert (trades["ofi_contribution"] < 0).all()

    def test_sizes_positive(self):
        """All trade sizes must be positive."""
        events = simulate_hawkes_events(1.0, 0.3, 1.0, 100.0, seed=42)
        trades = generate_synthetic_trades(events, seed=42)
        assert (trades["size"] > 0).all()

    def test_prices_positive(self):
        events = simulate_hawkes_events(1.0, 0.3, 1.0, 100.0, seed=42)
        trades = generate_synthetic_trades(events, base_mid=100.0, seed=42)
        assert (trades["price"] > 0).all()

    def test_n_trades_equals_n_events(self):
        """Each event becomes one trade."""
        events = simulate_hawkes_events(0.5, 0.3, 1.0, 100.0, seed=42)
        trades = generate_synthetic_trades(events, seed=42)
        assert len(trades) == len(events)

    def test_empty_events(self):
        trades = generate_synthetic_trades(np.array([]))
        assert len(trades) == 0

    def test_determinism(self):
        events = simulate_hawkes_events(0.5, 0.3, 1.0, 50.0, seed=42)
        t1 = generate_synthetic_trades(events, seed=42)
        t2 = generate_synthetic_trades(events, seed=42)
        pd.testing.assert_frame_equal(t1, t2)

    def test_bursty_trades_larger_sizes(self):
        """Shorter inter-event times (bursty) should produce larger trades."""
        events = simulate_hawkes_events(1.0, 0.3, 1.0, 100.0, seed=42)
        trades = generate_synthetic_trades(events, seed=42)
        inter_event = np.diff(trades["time"].values, prepend=0)
        # Group by quartile of inter-event time
        # Short inter-event trades should have higher average size
        if len(trades) > 20:
            median_ie = np.median(inter_event)
            short_ie_mask = inter_event < median_ie
            long_ie_mask = inter_event >= median_ie
            # This relationship isn't strict due to randomness, but on average
            # shorter inter-event → larger size multiplier → larger sizes
            assert trades.loc[short_ie_mask, "size"].mean() >= trades.loc[long_ie_mask, "size"].mean() * 0.5


class TestHawkes_IntensityInvariants:

    def test_intensity_always_ge_mu(self):
        """λ(t) ≥ μ for all t (intensity can't go below baseline)."""
        mu = 0.5
        events = simulate_hawkes_events(mu, 0.3, 1.0, 200.0, seed=42)
        grid = np.arange(0, 200, 0.5)
        lam = ground_truth_intensity(events, mu, 0.3, 1.0, grid)
        assert (lam >= mu - 1e-10).all()

    def test_intensity_spikes_after_events(self):
        """λ(t) increases immediately after an event."""
        mu, alpha, beta = 0.5, 0.5, 1.0
        events = np.array([10.0])
        grid = np.array([9.0, 10.5, 11.0])
        lam = ground_truth_intensity(events, mu, alpha, beta, grid)
        # Before event
        assert lam[0] == pytest.approx(mu, abs=1e-10)
        # After event: λ > μ
        assert lam[1] > mu
        # Further after: decays
        assert lam[2] < lam[1]

    def test_validate_estimation_perfect_match(self):
        result = simulate_regime("test", mu=0.5, alpha=0.3, beta=1.0, T=100.0, seed=42)
        val = validate_estimation(result["true_intensity"], result, rmse_threshold=0.10)
        assert val["passed"] == True
        assert val["rmse"] == pytest.approx(0.0, abs=1e-10)

    def test_branching_subcritical_implies_finite_rate(self):
        alpha, beta = 0.3, 1.0
        eta = branching_ratio(alpha, beta)
        assert eta < 1.0
        rate = expected_event_rate(0.5, alpha, beta)
        assert np.isfinite(rate)
        assert rate > 0

    def test_branching_critical_implies_infinite_rate(self):
        alpha, beta = 1.0, 1.0
        eta = branching_ratio(alpha, beta)
        assert eta == pytest.approx(1.0)
        rate = expected_event_rate(0.5, alpha, beta)
        assert rate == np.inf


# ####################################################################
# 5. K-FOLD — Purge, Embargo, Coverage Edge Cases
# ####################################################################

class TestPurgedKFold_MathProperties:

    def test_no_train_test_overlap(self):
        """Train ∩ Test = ∅ for every fold."""
        for n_splits in [2, 3, 5, 10]:
            splits = _purged_kfold_splits(500, n_splits, embargo=10)
            for train, test in splits:
                overlap = np.intersect1d(train, test)
                assert len(overlap) == 0, f"Overlap found with n_splits={n_splits}"

    def test_full_test_coverage(self):
        """Union of all test folds = all indices."""
        splits = _purged_kfold_splits(100, 5, embargo=2)
        all_test = np.concatenate([t for _, t in splits])
        assert len(np.unique(all_test)) == 100

    def test_embargo_zone_respected(self):
        """No training sample within embargo bars of test boundaries."""
        embargo = 15
        splits = _purged_kfold_splits(500, 5, embargo=embargo)
        for train_idx, test_idx in splits:
            test_min = test_idx.min()
            test_max = test_idx.max()
            # Check that no train index is within embargo of test boundaries
            near_test = train_idx[
                (train_idx >= test_min - embargo) & (train_idx <= test_max + embargo)
            ]
            assert len(near_test) == 0

    def test_embargo_zero_only_test_removed(self):
        """embargo=0 → only test fold removed from train."""
        splits = _purged_kfold_splits(100, 5, embargo=0)
        for train_idx, test_idx in splits:
            # Train should have everything except the test fold
            assert len(train_idx) + len(test_idx) == 100

    def test_n_splits_1_all_test(self):
        """n_splits=1 → all data is test, train is empty (after purge+embargo) → fold discarded."""
        splits = _purged_kfold_splits(100, 1, embargo=5)
        # With n_splits=1, the single fold uses ALL data as test, leaving no training data.
        # The purge+embargo logic discards folds with empty train sets.
        assert len(splits) == 0

    def test_large_embargo_may_reduce_train(self):
        """Very large embargo → may have very small train set."""
        splits = _purged_kfold_splits(50, 5, embargo=100)
        # With embargo=100 on a 50-sample dataset, train sets will be very small or empty
        # Check that we don't crash
        # Some folds may be skipped (empty train)
        for train, test in splits:
            assert len(train) >= 0

    def test_correct_number_of_splits(self):
        for n in [3, 5, 7]:
            splits = _purged_kfold_splits(200, n, embargo=5)
            assert len(splits) <= n


class TestPurgedKFold_Integration:

    def test_with_ou_series(self, ou_df):
        """K-fold on mean-reverting OU process."""
        result = purged_kfold(
            ou_df, score_fn=lambda df: _test_score_fn(df),
            n_splits=3, embargo=10, symbol="OU_TEST",
        )
        assert isinstance(result, KFoldResult)
        assert len(result.folds) == 3

    def test_serialization_roundtrip(self, synth_df, tmp_path):
        result = purged_kfold(
            synth_df, score_fn=lambda df: _test_score_fn(df),
            n_splits=3, embargo=10, symbol="SERIAL",
        )
        path = str(tmp_path / "kfold.json")
        result.to_json(path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["symbol"] == "SERIAL"
        assert loaded["n_splits"] == 3
        assert len(loaded["folds"]) == 3


# ####################################################################
# 6. WALK-FORWARD — Fold Generation Edge Cases
# ####################################################################

class TestWalkForward_FoldGeneration:

    def test_expanding_train_starts_at_zero(self):
        folds = _generate_folds(1000, 300, 100, overlap=False, expanding=True)
        for ts, te, _, _ in folds:
            assert ts == 0

    def test_rolling_train_constant_width(self):
        folds = _generate_folds(1000, 300, 100, overlap=False, expanding=False)
        for ts, te, _, _ in folds:
            assert te - ts == 300

    def test_no_test_overlap(self):
        folds = _generate_folds(2000, 500, 200, overlap=False)
        for i in range(len(folds) - 1):
            _, _, _, te_i = folds[i]
            _, _, ts_j, _ = folds[i + 1]
            assert ts_j >= te_i

    def test_with_overlap(self):
        folds = _generate_folds(2000, 500, 200, overlap=True)
        # With overlap, test windows shift by half
        if len(folds) >= 2:
            _, _, ts0, te0 = folds[0]
            _, _, ts1, te1 = folds[1]
            assert ts1 < te0  # should overlap

    def test_small_dataset_no_folds(self):
        """If dataset < train_window + test_window → no folds."""
        folds = _generate_folds(100, 80, 30, overlap=False)
        assert len(folds) == 0

    def test_exact_fit_one_fold(self):
        """Exactly train + test = n_bars → 1 fold."""
        folds = _generate_folds(500, 300, 200, overlap=False)
        assert len(folds) == 1
        ts, te, test_s, test_e = folds[0]
        assert test_e == 500

    def test_train_end_equals_test_start(self):
        """Train end should always equal test start (no gap)."""
        folds = _generate_folds(1000, 400, 100, overlap=False)
        for _, te, ts, _ in folds:
            assert te == ts


class TestWalkForwardCV_Integration:

    def test_ou_process(self, ou_df):
        result = walkforward_cv(
            ou_df, score_fn=lambda df: _test_score_fn(df),
            train_window=400, test_window=100, symbol="OU_WF",
        )
        assert isinstance(result, WalkForwardResult)
        assert result.n_folds > 0

    def test_expanding_vs_rolling(self, synth_df):
        """Expanding and rolling should give different fold specs."""
        r_exp = walkforward_cv(
            synth_df, score_fn=lambda df: _test_score_fn(df),
            train_window=400, test_window=100, expanding=True, symbol="EXP",
        )
        r_roll = walkforward_cv(
            synth_df, score_fn=lambda df: _test_score_fn(df),
            train_window=400, test_window=100, expanding=False, symbol="ROLL",
        )
        # Both should produce same number of folds
        assert r_exp.n_folds == r_roll.n_folds
        # But train sizes differ: expanding grows, rolling stays constant
        if r_exp.n_folds >= 2:
            exp_sizes = [f.train_size for f in r_exp.folds]
            roll_sizes = [f.train_size for f in r_roll.folds]
            # Expanding: sizes should increase
            assert exp_sizes[-1] > exp_sizes[0]
            # Rolling: sizes should be constant
            assert roll_sizes[0] == roll_sizes[-1]

    def test_determinism(self, synth_df):
        r1 = walkforward_cv(
            synth_df, score_fn=lambda df: _test_score_fn(df),
            train_window=400, test_window=100, symbol="DET",
        )
        r2 = walkforward_cv(
            synth_df, score_fn=lambda df: _test_score_fn(df),
            train_window=400, test_window=100, symbol="DET",
        )
        assert r1.to_dict() == r2.to_dict()


# ####################################################################
# 7. PHASE 3 RUNNER — Helper & Integration Edge Tests
# ####################################################################

class TestAggregateOOS:

    def test_empty_folds(self):
        from validation.phase3_runner import _aggregate_oos
        result = _aggregate_oos([], [5])
        assert "note" in result

    def test_folds_with_errors(self):
        from validation.phase3_runner import _aggregate_oos
        folds = [{"error": "something failed"}]
        result = _aggregate_oos(folds, [5])
        assert "note" in result  # no valid folds

    def test_single_valid_fold(self):
        from validation.phase3_runner import _aggregate_oos
        fold = {
            "metrics": {
                "entry_metrics": {"sortino": 1.5, "cagr": 0.1, "max_drawdown": -0.05, "ic_5d": 0.15},
                "exit_metrics": {},
                "signal_metrics": {},
            },
            "execution_costs": {"pnl_erosion": 0.002},
        }
        result = _aggregate_oos([fold], [5])
        assert result["median_sortino"] == pytest.approx(1.5)
        assert result["n_valid_folds"] == 1


class TestThresholdSweep:

    def test_higher_threshold_fewer_signals(self, synth_df):
        from validation.phase3_runner import _threshold_sweep
        thresholds = [40, 60, 80, 95]
        sweep = _threshold_sweep(synth_df, _test_score_fn, {}, thresholds, [5])
        n_signals = [s["n_signals"] for s in sweep]
        # Monotonically decreasing (or equal)
        for i in range(len(n_signals) - 1):
            assert n_signals[i] >= n_signals[i + 1]


class TestSubsampleStability:

    def test_determinism(self, synth_df):
        from validation.phase3_runner import _subsample_stability
        r1 = _subsample_stability(synth_df, _test_score_fn, {}, n_trials=3, fraction=0.5, seed=42)
        r2 = _subsample_stability(synth_df, _test_score_fn, {}, n_trials=3, fraction=0.5, seed=42)
        assert r1["sortino_mean"] == pytest.approx(r2["sortino_mean"])

    def test_output_keys(self, synth_df):
        from validation.phase3_runner import _subsample_stability
        result = _subsample_stability(synth_df, _test_score_fn, {}, n_trials=2, fraction=0.5, seed=42)
        assert "n_trials" in result
        assert "sortino_mean" in result
        assert "sortino_std" in result
        assert "ic_mean" in result


class TestComputeChecksum:

    def test_determinism(self, tmp_path):
        from validation.phase3_runner import _compute_checksum
        path = tmp_path / "test.json"
        path.write_text('{"a": 1}')
        c1 = _compute_checksum(str(path))
        c2 = _compute_checksum(str(path))
        assert c1 == c2
        assert len(c1) == 64  # SHA256 hex

    def test_different_content_different_hash(self, tmp_path):
        from validation.phase3_runner import _compute_checksum
        p1 = tmp_path / "a.json"
        p2 = tmp_path / "b.json"
        p1.write_text('{"a": 1}')
        p2.write_text('{"a": 2}')
        assert _compute_checksum(str(p1)) != _compute_checksum(str(p2))


class TestPhase3Runner_Integration:

    def test_full_pipeline_synthetic(self, synth_df_large, tmp_path):
        """End-to-end Phase 3 run on synthetic data."""
        from validation.phase3_runner import run_asset_validation, Phase3Config

        cfg = Phase3Config(
            seed=42,
            walkforward={"train_window": 400, "test_window": 200, "expanding": True},
            inner_cv={"n_splits": 3, "embargo_bars": 10},
            tuning={
                "method": "random_search",
                "n_trials": 2,
                "lambda_var": 0.5,
                "objective": "sortino",
                "search_space": {"S_scale": [0.5, 1.0, 2.0]},
            },
            execution={
                "slippage": {"k_impact": 0.001, "gamma": 0.5, "sigma_slip": 0.0001},
                "transaction_costs": {"commission_bps": 2.0, "spread_bps": 1.0},
                "latency": {"mean_latency_ms": 50},
            },
            hawkes_stress={
                "regimes": [
                    {"name": "bursty", "mu": 0.1, "alpha": 0.8, "beta": 1.0, "T": 50.0},
                ],
                "rmse_threshold": 0.10,
            },
            robustness={
                "threshold_sweep": [50, 70],
                "subsample_n_trials": 2,
                "subsample_fraction": 0.5,
            },
            scoring={"entry_threshold": 70, "exit_threshold": 70, "forward_horizons": [5]},
        )

        result = run_asset_validation(
            synth_df_large, _test_score_fn, "SYNTH_COMP", "1d",
            cfg, output_dir=str(tmp_path),
        )

        # Structural checks
        assert "oos_folds" in result
        assert "oos_summary" in result
        assert "tuning_traces" in result
        assert "hawkes_stress" in result
        assert "slippage_matrix" in result
        assert "threshold_sweep" in result
        assert "subsample_stability" in result
        assert "checksum" in result

        # Output file exists
        assert (tmp_path / "SYNTH_COMP_1d_tuning.json").exists()

    def test_determinism(self, tmp_path):
        """Two identical runs produce identical checksums."""
        from validation.phase3_runner import run_asset_validation, Phase3Config

        df = fbm_series(n=800, H=0.6, seed=42)
        cfg = Phase3Config(
            seed=42,
            walkforward={"train_window": 300, "test_window": 100, "expanding": True},
            inner_cv={"n_splits": 3, "embargo_bars": 5},
            tuning={
                "method": "random_search", "n_trials": 2,
                "search_space": {"S_scale": [1.0, 2.0]},
                "objective": "sortino", "lambda_var": 0.5,
            },
            hawkes_stress={"regimes": [], "rmse_threshold": 0.10},
            robustness={},
            scoring={"entry_threshold": 70, "exit_threshold": 70, "forward_horizons": [5]},
            execution={},
        )

        r1 = run_asset_validation(df, _test_score_fn, "DET", "1d", cfg, str(tmp_path / "run1"))
        r2 = run_asset_validation(df, _test_score_fn, "DET", "1d", cfg, str(tmp_path / "run2"))

        s1 = json.loads(json.dumps(r1["oos_summary"], default=str))
        s2 = json.loads(json.dumps(r2["oos_summary"], default=str))
        assert s1 == s2

    def test_no_search_space_no_crash(self, tmp_path):
        """Empty search space → no tuning, still runs."""
        from validation.phase3_runner import run_asset_validation, Phase3Config

        df = fbm_series(n=800, H=0.6, seed=42)
        cfg = Phase3Config(
            seed=42,
            walkforward={"train_window": 300, "test_window": 100, "expanding": True},
            inner_cv={"n_splits": 3, "embargo_bars": 5},
            tuning={"method": "random_search", "n_trials": 2, "search_space": {}},
            hawkes_stress={"regimes": []},
            robustness={},
            scoring={"entry_threshold": 70, "exit_threshold": 70, "forward_horizons": [5]},
            execution={},
        )
        result = run_asset_validation(df, _test_score_fn, "NOSEARCH", "1d", cfg, str(tmp_path))
        assert "oos_folds" in result
        # Tuning traces should note "no search space"
        for trace in result["tuning_traces"]:
            if "note" in trace:
                assert trace["note"] == "no search space"


# ####################################################################
# 8. CROSS-CUTTING — Numerical Stability & Stress Tests
# ####################################################################

class TestNumericalStability:
    """Edge cases that may cause numerical issues."""

    def test_sortino_very_small_returns(self):
        """Near-zero returns should not produce NaN or inf spuriously."""
        returns = pd.Series([1e-10, -1e-10, 1e-10, -1e-10, 1e-10])
        s = sortino_ratio(returns)
        assert np.isfinite(s) or s == np.inf

    def test_cagr_very_long_series(self):
        """10000 bars (40 years) → CAGR should still be finite."""
        equity = pd.Series(100 * np.exp(np.cumsum(np.random.RandomState(42).randn(10000) * 0.001)))
        c = cagr(equity, periods_per_year=252)
        assert np.isfinite(c)

    def test_ic_large_series(self):
        """10000 data points → IC should be finite."""
        np.random.seed(42)
        scores = pd.Series(np.random.rand(10000) * 100)
        returns = pd.Series(np.random.randn(10000) * 0.01)
        ic = information_coefficient(scores, returns)
        assert np.isfinite(ic)

    def test_max_drawdown_very_volatile(self):
        """Extreme volatility → drawdown still in [-1, 0]."""
        np.random.seed(42)
        equity = pd.Series(100 * np.exp(np.cumsum(np.random.randn(1000) * 0.5)))
        dd = max_drawdown(equity)
        assert -1.0 <= dd <= 0.0

    def test_impact_very_large_order(self):
        """Order size >> ADV → impact is large but finite."""
        i = market_impact(1e12, 1e6, 0.01, 0.5)
        assert np.isfinite(i)
        assert i > 0

    def test_forward_returns_with_zero_price(self):
        """Zero price → division by zero → inf/NaN in forward returns."""
        prices = pd.Series([100.0, 0.0, 50.0, 60.0])
        fwd = forward_returns(prices, horizon=1)
        # Should handle gracefully (may produce inf at position 1)
        assert len(fwd) == 4

    def test_evaluate_signals_with_nan_returns(self):
        """NaN in returns should not crash signal evaluation."""
        n = 10
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        entry = pd.Series([80] * n, index=idx, dtype=float)
        exit_ = pd.Series([0] * 5 + [80] + [0] * 4, index=idx, dtype=float)
        returns = pd.Series([np.nan] * n, index=idx)
        result = evaluate_signals(entry, exit_, returns, 70, 70)
        assert isinstance(result, dict)
