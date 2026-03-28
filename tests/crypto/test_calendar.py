"""Tests for crypto.calendar."""

import numpy as np
import pandas as pd
import pytest

from engine.crypto.calendar import (
    ANNUALIZATION_FACTORS,
    annualized_cagr,
    annualized_sharpe,
    annualized_sortino,
    bars_per_day,
    calmar_ratio,
    get_annualization_factor,
    max_drawdown,
    profit_factor,
)


class TestAnnualizationFactors:
    def test_known_values(self):
        assert get_annualization_factor("1h") == 8_760
        assert get_annualization_factor("1d") == 365
        assert get_annualization_factor("5m") == 105_120

    def test_invalid_timeframe(self):
        with pytest.raises(ValueError):
            get_annualization_factor("2h")

    def test_bars_per_day(self):
        assert bars_per_day("1h") == pytest.approx(24.0)
        assert bars_per_day("1d") == pytest.approx(1.0)
        assert bars_per_day("4h") == pytest.approx(6.0)


class TestAnnualizedSharpe:
    def test_positive_returns(self):
        returns = pd.Series(np.random.RandomState(42).normal(0.001, 0.02, 1000))
        sharpe = annualized_sharpe(returns, "1h")
        assert sharpe > 0

    def test_zero_std(self):
        returns = pd.Series([0.0, 0.0, 0.0])
        assert annualized_sharpe(returns, "1h") == 0.0

    def test_too_few_returns(self):
        returns = pd.Series([0.01])
        assert annualized_sharpe(returns, "1h") == 0.0

    def test_different_timeframes_scale(self):
        returns = pd.Series(np.random.RandomState(42).normal(0.001, 0.02, 500))
        sharpe_1h = annualized_sharpe(returns, "1h")
        sharpe_1d = annualized_sharpe(returns, "1d")
        assert sharpe_1h != sharpe_1d


class TestAnnualizedSortino:
    def test_positive_returns(self):
        returns = pd.Series(np.random.RandomState(42).normal(0.001, 0.02, 1000))
        sortino = annualized_sortino(returns, "1h")
        assert sortino > 0

    def test_no_downside(self):
        returns = pd.Series([0.01, 0.02, 0.03])
        sortino = annualized_sortino(returns, "1h")
        assert sortino == pytest.approx(50.0)


class TestAnnualizedCagr:
    def test_doubling(self):
        equity = pd.Series([100.0, 200.0])
        cagr = annualized_cagr(equity, "1d")
        assert cagr == pytest.approx(1.0)

    def test_flat(self):
        equity = pd.Series([100.0] * 365)
        cagr = annualized_cagr(equity, "1d")
        assert cagr == pytest.approx(0.0)

    def test_decline(self):
        equity = pd.Series([100.0, 90.0, 80.0])
        cagr = annualized_cagr(equity, "1d")
        assert cagr < 0

    def test_empty(self):
        assert annualized_cagr(pd.Series([100.0]), "1d") == 0.0


class TestMaxDrawdown:
    def test_no_drawdown(self):
        equity = pd.Series([100, 110, 120, 130])
        assert max_drawdown(equity) == pytest.approx(0.0)

    def test_known_drawdown(self):
        equity = pd.Series([100.0, 120.0, 90.0, 110.0])
        mdd = max_drawdown(equity)
        assert mdd == pytest.approx(-0.25, rel=0.01)

    def test_empty(self):
        assert max_drawdown(pd.Series([100.0])) == 0.0


class TestCalmarRatio:
    def test_positive_calmar(self):
        equity = pd.Series(np.linspace(100, 200, 365))
        cr = calmar_ratio(equity, "1d")
        assert cr > 0

    def test_zero_drawdown(self):
        equity = pd.Series([100, 110, 120])
        cr = calmar_ratio(equity, "1d")
        assert cr > 0


class TestProfitFactor:
    def test_balanced(self):
        trades = pd.Series([100, -50, 80, -30])
        pf = profit_factor(trades)
        assert pf == pytest.approx(180.0 / 80.0)

    def test_all_winners(self):
        trades = pd.Series([100, 50, 80])
        pf = profit_factor(trades)
        assert pf == float("inf")

    def test_all_losers(self):
        trades = pd.Series([-100, -50, -80])
        pf = profit_factor(trades)
        assert pf == pytest.approx(0.0)
