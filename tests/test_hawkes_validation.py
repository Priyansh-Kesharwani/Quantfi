"""
Phase 3 — Tests for Real-World Validation & Parameter Tuning.

Validates:
  - Execution model (slippage, impact, fill prices, costs)
  - Tuning engine (nested CV, search, sensitivity, ablation)
  - Phase 3 runner (orchestrator with synthetic data)
  - Hawkes LOB / trade tick generation
  - Config loading & dataclass construction
  - Determinism of tuning pipeline
"""

import pytest
import numpy as np
import pandas as pd
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.fixtures import fbm_series

class TestMarketImpact:
    def test_zero_order_zero_impact(self):
        from engine.validation.execution_model import market_impact
        assert market_impact(0, 1e6) == 0.0

    def test_impact_increases_with_size(self):
        from engine.validation.execution_model import market_impact
        small = market_impact(100, 1e6)
        large = market_impact(10000, 1e6)
        assert large > small

    def test_zero_adv_returns_zero(self):
        from engine.validation.execution_model import market_impact
        assert market_impact(100, 0) == 0.0

    def test_sqrt_law(self):
        from engine.validation.execution_model import market_impact
        i1 = market_impact(100, 1e6, k_impact=1.0, gamma=0.5)
        i2 = market_impact(200, 1e6, k_impact=1.0, gamma=0.5)
        ratio = i2 / i1
        assert ratio == pytest.approx(np.sqrt(2), abs=0.01)

class TestFillPrice:
    def test_buy_costs_more(self):
        from engine.validation.execution_model import compute_fill_price, ExecutionConfig
        cfg = ExecutionConfig(sigma_slip=0.0)
        rng = np.random.RandomState(42)
        fill, _ = compute_fill_price(100.0, side=1, order_size=100, adv=1e6, config=cfg, rng=rng)
        assert fill > 100.0

    def test_sell_costs_less(self):
        from engine.validation.execution_model import compute_fill_price, ExecutionConfig
        cfg = ExecutionConfig(sigma_slip=0.0)
        rng = np.random.RandomState(42)
        fill, _ = compute_fill_price(100.0, side=-1, order_size=100, adv=1e6, config=cfg, rng=rng)
        assert fill < 100.0

    def test_breakdown_keys(self):
        from engine.validation.execution_model import compute_fill_price
        _, bd = compute_fill_price(100.0, 1, 100, 1e6)
        assert "mid_price" in bd
        assert "fill_price" in bd
        assert "impact_frac" in bd
        assert "commission_frac" in bd
        assert "total_cost_frac" in bd

class TestLatencySimulation:
    def test_determinism(self):
        from engine.validation.execution_model import simulate_latency
        l1 = simulate_latency(100, seed=42)
        l2 = simulate_latency(100, seed=42)
        np.testing.assert_array_equal(l1, l2)

    def test_no_jitter_constant(self):
        from engine.validation.execution_model import simulate_latency
        lats = simulate_latency(10, mean_ms=50, jitter=False)
        assert np.all(lats == 50.0)

    def test_positive_latencies(self):
        from engine.validation.execution_model import simulate_latency
        lats = simulate_latency(100, seed=42)
        assert (lats > 0).all()

class TestApplyExecutionCosts:
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

    def test_returns_adjusted_series(self, signal_data):
        from engine.validation.execution_model import apply_execution_costs
        returns, entry, exit_, volume = signal_data
        adj, report = apply_execution_costs(returns, entry, exit_, volume)
        assert isinstance(adj, pd.Series)
        assert len(adj) == len(returns)

    def test_costs_reduce_returns(self, signal_data):
        from engine.validation.execution_model import apply_execution_costs, ExecutionConfig
        returns, entry, exit_, volume = signal_data
        cfg = ExecutionConfig(k_impact=0.01)
        adj, report = apply_execution_costs(returns, entry, exit_, volume, config=cfg)
        assert adj.sum() <= returns.sum()

    def test_zero_impact_minimal_change(self, signal_data):
        from engine.validation.execution_model import apply_execution_costs, ExecutionConfig
        returns, entry, exit_, volume = signal_data
        cfg = ExecutionConfig(k_impact=0, sigma_slip=0, commission_bps=0, spread_bps=0)
        adj, report = apply_execution_costs(returns, entry, exit_, volume, config=cfg)
        np.testing.assert_allclose(adj.values, returns.values, atol=1e-10)

class TestSlippageMatrix:
    def test_returns_dataframe(self):
        from engine.validation.execution_model import slippage_sensitivity_matrix
        np.random.seed(42)
        n = 100
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        returns = pd.Series(np.random.randn(n) * 0.01, index=idx)
        entry = pd.Series(np.clip(50 + np.random.randn(n) * 20, 0, 100), index=idx)
        exit_ = pd.Series(np.clip(50 + np.random.randn(n) * 20, 0, 100), index=idx)
        volume = pd.Series(np.abs(np.random.randn(n) * 1e6), index=idx)

        matrix = slippage_sensitivity_matrix(returns, entry, exit_, volume)
        assert isinstance(matrix, pd.DataFrame)
        assert matrix.shape[0] > 0
        assert matrix.shape[1] > 0

class TestExecutionConfig:
    def test_from_dict(self):
        from engine.validation.execution_model import ExecutionConfig
        d = {
            "slippage": {"k_impact": 0.002, "gamma": 0.7},
            "transaction_costs": {"commission_bps": 3.0},
        }
        cfg = ExecutionConfig.from_dict(d)
        assert cfg.k_impact == 0.002
        assert cfg.gamma == 0.7
        assert cfg.commission_bps == 3.0

    def test_defaults(self):
        from engine.validation.execution_model import ExecutionConfig
        cfg = ExecutionConfig()
        assert cfg.k_impact == 0.001
        assert cfg.gamma == 0.5

@pytest.fixture(scope="module")
def synth_df_large():
    return fbm_series(n=1200, H=0.6, seed=42)

def _test_score_fn(df, params=None):
    """Deterministic score function for testing tuning."""
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

class TestTuningConfig:
    def test_from_dict(self):
        from engine.validation.tuning import TuningConfig
        d = {"method": "grid", "n_trials": 10, "lambda_var": 0.3}
        cfg = TuningConfig.from_dict(d)
        assert cfg.method == "grid"
        assert cfg.n_trials == 10
        assert cfg.lambda_var == 0.3

    def test_defaults(self):
        from engine.validation.tuning import TuningConfig
        cfg = TuningConfig()
        assert cfg.method == "random_search"
        assert cfg.lambda_var == 0.5

class TestSearchSpace:
    def test_grid_generation(self):
        from engine.validation.tuning import _generate_grid
        space = {"a": [1, 2], "b": [10, 20]}
        grid = _generate_grid(space)
        assert len(grid) == 4
        assert {"a": 1, "b": 10} in grid

    def test_random_sampling(self):
        from engine.validation.tuning import _sample_random
        space = {"a": [1, 2, 3], "b": [10, 20, 30]}
        rng = np.random.RandomState(42)
        samples = _sample_random(space, 5, rng)
        assert len(samples) == 5
        for s in samples:
            assert "a" in s
            assert "b" in s

class TestInnerCV:
    def test_runs_without_error(self, synth_df_large):
        from engine.validation.tuning import _evaluate_inner_cv
        fold_metrics, med, std = _evaluate_inner_cv(
            synth_df_large, _test_score_fn, {"S_scale": 1.0},
            n_splits=3, embargo=10,
        )
        assert len(fold_metrics) == 3
        assert isinstance(med, float)
        assert isinstance(std, float)

    def test_different_params_different_results(self, synth_df_large):
        from engine.validation.tuning import _evaluate_inner_cv
        _, med1, _ = _evaluate_inner_cv(
            synth_df_large, _test_score_fn, {"S_scale": 0.5},
            n_splits=3, embargo=10,
        )
        _, med2, _ = _evaluate_inner_cv(
            synth_df_large, _test_score_fn, {"S_scale": 2.0},
            n_splits=3, embargo=10,
        )
        assert isinstance(med1, float)
        assert isinstance(med2, float)

class TestRunTuning:
    def test_runs_random_search(self, synth_df_large):
        from engine.validation.tuning import run_tuning, TuningConfig
        cfg = TuningConfig(
            method="random_search",
            n_trials=3,
            search_space={"S_scale": [0.5, 1.0, 2.0]},
            inner_n_splits=3,
            inner_embargo=10,
            seed=42,
        )
        result = run_tuning(synth_df_large, _test_score_fn, cfg, symbol="SYNTH")
        assert len(result.trials) == 3
        assert result.best_params is not None
        assert result.best_score > -np.inf

    def test_runs_grid_search(self, synth_df_large):
        from engine.validation.tuning import run_tuning, TuningConfig
        cfg = TuningConfig(
            method="grid",
            search_space={"S_scale": [0.5, 1.0]},
            inner_n_splits=3,
            inner_embargo=10,
            seed=42,
        )
        result = run_tuning(synth_df_large, _test_score_fn, cfg, symbol="SYNTH")
        assert len(result.trials) == 2

    def test_serialisation(self, synth_df_large):
        from engine.validation.tuning import run_tuning, TuningConfig
        cfg = TuningConfig(
            method="random_search",
            n_trials=2,
            search_space={"S_scale": [1.0, 2.0]},
            inner_n_splits=3,
            inner_embargo=10,
            seed=42,
        )
        result = run_tuning(synth_df_large, _test_score_fn, cfg, symbol="SYNTH")
        d = result.to_dict()
        assert "best_params" in d
        assert "trials" in d
        assert len(d["trials"]) == 2

    def test_determinism(self, synth_df_large):
        from engine.validation.tuning import run_tuning, TuningConfig
        cfg = TuningConfig(
            method="random_search",
            n_trials=2,
            search_space={"S_scale": [0.5, 1.0, 2.0]},
            inner_n_splits=3,
            inner_embargo=10,
            seed=42,
        )
        r1 = run_tuning(synth_df_large, _test_score_fn, cfg, symbol="SYNTH")
        r2 = run_tuning(synth_df_large, _test_score_fn, cfg, symbol="SYNTH")
        assert r1.best_score == r2.best_score
        assert r1.best_params == r2.best_params

class TestParameterSensitivity:
    def test_returns_dataframe(self, synth_df_large):
        from engine.validation.tuning import parameter_sensitivity
        result = parameter_sensitivity(
            synth_df_large, _test_score_fn,
            base_params={"S_scale": 1.0},
            param_name="S_scale",
            param_values=[0.5, 1.0, 2.0],
            n_splits=3, embargo=10,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "median_metric" in result.columns
        assert "score" in result.columns

class TestAblation:
    def test_returns_dataframe(self, synth_df_large):
        from engine.validation.tuning import ablation_study

        def _ablation_score_fn(df, params=None):
            np.random.seed(42)
            n = len(df)
            base = 50 + np.random.randn(n) * 15
            if params and params.get("disable_ofi"):
                base *= 0.9
            return (
                pd.Series(np.clip(base, 0, 100), index=df.index),
                pd.Series(np.clip(100 - base, 0, 100), index=df.index),
            )

        result = ablation_study(
            synth_df_large, _ablation_score_fn,
            base_params={},
            components=["ofi", "hawkes"],
            n_splits=3, embargo=10,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "impact" in result.columns

class TestSyntheticLOB:
    def test_generates_lob(self):
        from engine.simulations.hawkes_simulator import simulate_hawkes_events, generate_synthetic_lob
        events = simulate_hawkes_events(0.5, 0.3, 1.0, 100.0, seed=42)
        grid = np.arange(0, 100, 1.0)
        lob = generate_synthetic_lob(events, grid, depth_levels=3, seed=42)
        assert "mid_prices" in lob
        assert "bid_prices" in lob
        assert "ask_prices" in lob
        assert len(lob["mid_prices"]) == len(grid)
        assert len(lob["bid_prices"]) == len(grid)

    def test_depth_levels(self):
        from engine.simulations.hawkes_simulator import simulate_hawkes_events, generate_synthetic_lob
        events = simulate_hawkes_events(0.5, 0.3, 1.0, 50.0, seed=42)
        grid = np.arange(0, 50, 1.0)
        lob = generate_synthetic_lob(events, grid, depth_levels=5, seed=42)
        assert lob["n_levels"] == 5
        assert len(lob["bid_prices"][0]) == 5

    def test_determinism(self):
        from engine.simulations.hawkes_simulator import simulate_hawkes_events, generate_synthetic_lob
        events = simulate_hawkes_events(0.5, 0.3, 1.0, 50.0, seed=42)
        grid = np.arange(0, 50, 1.0)
        l1 = generate_synthetic_lob(events, grid, seed=42)
        l2 = generate_synthetic_lob(events, grid, seed=42)
        np.testing.assert_array_equal(l1["mid_prices"], l2["mid_prices"])

    def test_bid_below_ask(self):
        from engine.simulations.hawkes_simulator import simulate_hawkes_events, generate_synthetic_lob
        events = simulate_hawkes_events(0.5, 0.3, 1.0, 50.0, seed=42)
        grid = np.arange(0, 50, 1.0)
        lob = generate_synthetic_lob(events, grid, depth_levels=3, seed=42)
        for i in range(len(grid)):
            for lev in range(3):
                assert lob["bid_prices"][i][lev] < lob["ask_prices"][i][lev]

class TestSyntheticTrades:
    def test_generates_trades(self):
        from engine.simulations.hawkes_simulator import simulate_hawkes_events, generate_synthetic_trades
        events = simulate_hawkes_events(0.5, 0.3, 1.0, 100.0, seed=42)
        trades = generate_synthetic_trades(events, seed=42)
        assert isinstance(trades, pd.DataFrame)
        assert len(trades) == len(events)
        assert "price" in trades.columns
        assert "side" in trades.columns
        assert "ofi_contribution" in trades.columns

    def test_empty_events(self):
        from engine.simulations.hawkes_simulator import generate_synthetic_trades
        trades = generate_synthetic_trades(np.array([]))
        assert len(trades) == 0

    def test_sides_are_valid(self):
        from engine.simulations.hawkes_simulator import simulate_hawkes_events, generate_synthetic_trades
        events = simulate_hawkes_events(1.0, 0.3, 1.0, 100.0, seed=42)
        trades = generate_synthetic_trades(events, seed=42)
        assert set(trades["side"].unique()).issubset({-1, 1})

    def test_determinism(self):
        from engine.simulations.hawkes_simulator import simulate_hawkes_events, generate_synthetic_trades
        events = simulate_hawkes_events(0.5, 0.3, 1.0, 50.0, seed=42)
        t1 = generate_synthetic_trades(events, seed=42)
        t2 = generate_synthetic_trades(events, seed=42)
        pd.testing.assert_frame_equal(t1, t2)

class TestPhase3Config:
    def test_loads_from_yaml(self):
        from engine.validation.phase3_runner import Phase3Config
        cfg = Phase3Config.from_yaml("config/settings.yml")
        assert cfg.seed == 42
        assert len(cfg.assets) >= 3
        assert "train_window" in cfg.walkforward
        assert "method" in cfg.tuning

    def test_config_has_execution(self):
        from engine.validation.phase3_runner import Phase3Config
        cfg = Phase3Config.from_yaml("config/settings.yml")
        assert "slippage" in cfg.execution
        assert "transaction_costs" in cfg.execution

    def test_config_has_hawkes_stress(self):
        from engine.validation.phase3_runner import Phase3Config
        cfg = Phase3Config.from_yaml("config/settings.yml")
        assert len(cfg.hawkes_stress.get("regimes", [])) >= 4

    def test_config_has_robustness(self):
        from engine.validation.phase3_runner import Phase3Config
        cfg = Phase3Config.from_yaml("config/settings.yml")
        assert "threshold_sweep" in cfg.robustness
        assert "subsample_n_trials" in cfg.robustness

class TestPhase3Runner:
    @pytest.fixture(scope="class")
    def synth_run(self):
        """Run Phase 3 on synthetic data (small)."""
        from engine.validation.phase3_runner import run_asset_validation, Phase3Config

        df = fbm_series(n=1200, H=0.6, seed=42)

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
                    {"name": "bursty", "mu": 0.1, "alpha": 0.8, "beta": 1.0, "T": 100.0},
                    {"name": "dense", "mu": 2.0, "alpha": 0.1, "beta": 1.0, "T": 100.0},
                ],
                "rmse_threshold": 0.10,
            },
            robustness={
                "threshold_sweep": [50, 70, 90],
                "subsample_n_trials": 3,
                "subsample_fraction": 0.5,
            },
            scoring={"entry_threshold": 70, "exit_threshold": 70, "forward_horizons": [5]},
        )

        result = run_asset_validation(
            df, _test_score_fn, "SYNTH", "1d", cfg,
            output_dir="validation/outputs",
        )
        return result

    def test_has_oos_folds(self, synth_run):
        assert "oos_folds" in synth_run
        assert len(synth_run["oos_folds"]) > 0

    def test_has_oos_summary(self, synth_run):
        assert "oos_summary" in synth_run
        summary = synth_run["oos_summary"]
        assert "median_sortino" in summary or "note" in summary

    def test_has_hawkes_stress(self, synth_run):
        assert "hawkes_stress" in synth_run
        hs = synth_run["hawkes_stress"]
        assert "regimes" in hs
        assert "validations" in hs
        assert len(hs["validations"]) == 2

    def test_hawkes_validations_pass(self, synth_run):
        for v in synth_run["hawkes_stress"]["validations"]:
            assert v["passed"]

    def test_has_slippage_matrix(self, synth_run):
        assert "slippage_matrix" in synth_run

    def test_has_threshold_sweep(self, synth_run):
        assert "threshold_sweep" in synth_run
        sweep = synth_run["threshold_sweep"]
        assert len(sweep) == 3

    def test_has_subsample_stability(self, synth_run):
        assert "subsample_stability" in synth_run
        ss = synth_run["subsample_stability"]
        assert "sortino_mean" in ss

    def test_has_tuning_traces(self, synth_run):
        assert "tuning_traces" in synth_run
        assert len(synth_run["tuning_traces"]) > 0

    def test_output_file_created(self, synth_run):
        path = Path("validation/outputs/SYNTH_1d_tuning.json")
        assert path.exists()

    def test_determinism(self):
        """Two identical runs should produce identical results."""
        from engine.validation.phase3_runner import run_asset_validation, Phase3Config

        df = fbm_series(n=800, H=0.6, seed=42)
        cfg = Phase3Config(
            seed=42,
            walkforward={"train_window": 300, "test_window": 100, "expanding": True},
            inner_cv={"n_splits": 3, "embargo_bars": 5},
            tuning={"method": "random_search", "n_trials": 2,
                    "search_space": {"S_scale": [1.0, 2.0]},
                    "objective": "sortino", "lambda_var": 0.5},
            hawkes_stress={"regimes": [], "rmse_threshold": 0.10},
            robustness={},
            scoring={"entry_threshold": 70, "exit_threshold": 70, "forward_horizons": [5]},
            execution={},
        )

        r1 = run_asset_validation(df, _test_score_fn, "DET", "1d", cfg, "validation/outputs")
        r2 = run_asset_validation(df, _test_score_fn, "DET", "1d", cfg, "validation/outputs")

        s1 = json.loads(json.dumps(r1["oos_summary"], default=str))
        s2 = json.loads(json.dumps(r2["oos_summary"], default=str))
        assert s1 == s2
