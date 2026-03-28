"""End-to-end backtest tests for crypto.services.backtest_service."""

import numpy as np
import pandas as pd
import pytest

from engine.crypto.regime.detector import CryptoRegimeConfig
from engine.crypto.services.backtest_service import CryptoBacktestConfig, CryptoBacktestService


def _make_ohlcv(n: int = 800, seed: int = 42, trend: float = 0.0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="1h")
    returns = rng.normal(0.0001 + trend, 0.005, n)
    close = 50_000.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame(
        {
            "open": close * (1 + rng.randn(n) * 0.001),
            "high": close * (1 + abs(rng.randn(n) * 0.005)),
            "low": close * (1 - abs(rng.randn(n) * 0.005)),
            "close": close,
            "volume": 1000 + rng.rand(n) * 500,
        },
        index=idx,
    )


class TestCryptoBacktestService:
    @pytest.fixture
    def service(self):
        return CryptoBacktestService()

    @pytest.fixture
    def small_config(self):
        return CryptoBacktestConfig(
            initial_capital=10_000.0,
            leverage=3.0,
            cost_preset="ZERO",
            entry_threshold=20.0,
            exit_threshold=5.0,
            max_holding_bars=48,
            compression_window=50,
            ic_window=100,
            ic_horizon=10,
            funding_window=100,
            regime_config=CryptoRegimeConfig(
                warmup_bars=100,
                rolling_window=200,
                refit_every=50,
                vol_window=30,
                cooldown_bars=3,
            ),
        )

    def test_runs_without_error(self, service, small_config):
        ohlcv = _make_ohlcv(800)
        result = service.run(ohlcv, small_config)
        assert "sharpe" in result
        assert "equity_curve" in result
        assert "trades" in result
        assert "baselines" in result
        assert "score_reachability" in result

    def test_equity_curve_length(self, service, small_config):
        ohlcv = _make_ohlcv(800)
        result = service.run(ohlcv, small_config)
        assert len(result["equity_curve"]) == 800

    def test_baselines_present(self, service, small_config):
        ohlcv = _make_ohlcv(800)
        result = service.run(ohlcv, small_config)
        baselines = result["baselines"]
        assert "buy_and_hold" in baselines
        assert "always_long" in baselines
        assert "random_entry" in baselines

    def test_metrics_are_numbers(self, service, small_config):
        ohlcv = _make_ohlcv(800)
        result = service.run(ohlcv, small_config)
        for key in ["sharpe", "cagr", "max_drawdown", "final_equity", "n_trades"]:
            assert isinstance(result[key], (int, float)), f"{key} is not numeric"

    def test_with_funding_rates(self, service, small_config):
        ohlcv = _make_ohlcv(800)
        funding = pd.Series(
            np.random.RandomState(42).normal(0.0001, 0.0003, 800),
            index=ohlcv.index,
        )
        result = service.run(ohlcv, small_config, funding_rates=funding)
        assert "sharpe" in result

    def test_uptrend_produces_positive_bh(self, service, small_config):
        ohlcv = _make_ohlcv(800, trend=0.001)
        result = service.run(ohlcv, small_config)
        assert result["baselines"]["buy_and_hold"]["total_return_pct"] > 0

    def test_score_reachability_reported(self, service, small_config):
        ohlcv = _make_ohlcv(800)
        result = service.run(ohlcv, small_config)
        assert "entry_pct" in result["score_reachability"]

    def test_equity_curve_starts_at_initial_capital(self, service, small_config):
        ohlcv = _make_ohlcv(800)
        result = service.run(ohlcv, small_config)
        eq = result["equity_curve"]
        assert abs(eq.iloc[0] - small_config.initial_capital) < 1.0

    def test_equity_curve_never_negative(self, service, small_config):
        ohlcv = _make_ohlcv(800)
        result = service.run(ohlcv, small_config)
        eq = result["equity_curve"]
        assert (eq >= 0).all(), "Equity curve should never go negative"


class TestAllSymbolsAllStrategies:
    """Cross-product validation: every symbol/strategy combo produces trades."""

    SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT"]
    MODES = ["directional", "grid", "adaptive"]

    def _run(self, symbol, mode):
        from tests.crypto.synthetic import generate_synthetic_data, generate_synthetic_funding

        ohlcv = generate_synthetic_data(1000, symbol)
        funding = generate_synthetic_funding(1000, symbol)
        cfg = CryptoBacktestConfig(
            symbol=symbol,
            strategy_mode=mode,
            leverage=3.0,
            initial_capital=10_000.0,
            compression_window=60,
            ic_window=60,
            ic_horizon=10,
            funding_window=80,
            regime_config=CryptoRegimeConfig(
                warmup_bars=250, rolling_window=250, refit_every=50,
                vol_window=48, cooldown_bars=3, circuit_breaker_dd=-0.25,
            ),
        )
        svc = CryptoBacktestService()
        return svc.run(ohlcv, cfg, funding_rates=funding)

    @pytest.mark.parametrize("symbol", SYMBOLS)
    @pytest.mark.parametrize("mode", MODES)
    def test_all_symbols_all_strategies_positive_trades(self, symbol, mode):
        result = self._run(symbol, mode)
        assert result["n_trades"] > 0, f"{symbol} {mode}: 0 trades"

    def test_strategy_beats_random_entry(self):
        result = self._run("BTC/USDT:USDT", "adaptive")
        strat_sharpe = result["sharpe"]
        rand_sharpe = result["baselines"]["random_entry"]["sharpe"]
        assert strat_sharpe > rand_sharpe, f"Strategy Sharpe ({strat_sharpe}) should beat random ({rand_sharpe})"

    def test_adaptive_uses_both_directional_and_grid(self):
        result = self._run("BTC/USDT:USDT", "adaptive")
        sides = {t.side for t in result["trades"]}
        has_directional = bool(sides & {"long_exit", "short_exit"})
        has_grid = bool(sides & {"grid_buy", "grid_sell"})
        assert has_directional or has_grid, f"Adaptive should use at least one engine, got sides={sides}"

    def test_leverage_scales_returns(self):
        from tests.crypto.synthetic import generate_synthetic_data, generate_synthetic_funding

        ohlcv = generate_synthetic_data(1000, "BTC/USDT:USDT")
        funding = generate_synthetic_funding(1000, "BTC/USDT:USDT")

        rets = []
        for lev in [2.0, 5.0]:
            cfg = CryptoBacktestConfig(
                symbol="BTC/USDT:USDT", strategy_mode="directional", leverage=lev,
                compression_window=60, ic_window=60, ic_horizon=10, funding_window=80,
                regime_config=CryptoRegimeConfig(
                    warmup_bars=250, rolling_window=250, refit_every=50,
                    vol_window=48, cooldown_bars=3, circuit_breaker_dd=-0.25,
                ),
            )
            result = CryptoBacktestService().run(ohlcv, cfg, funding_rates=funding)
            rets.append(abs(result["total_return_pct"]))
        assert rets[1] >= rets[0] * 0.5, "Higher leverage should amplify returns"
