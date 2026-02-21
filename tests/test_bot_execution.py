"""
Unit tests for bot.execution: TBLManager, estimate_ou_params, var_future.

Synthetic price paths and OU series; barrier and formula checks.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bot.execution import TBLManager, estimate_ou_params, var_future


def test_tbl_manager_tp_hit():
    """Price at or above entry+tp triggers exit with reason 'tp'."""
    m = TBLManager(entry_price=100.0, tp=2.0, sl=1.5, tmax=10.0, entry_time=0.0)
    exit_flag, reason = m.on_tick(101.0, 1.0)
    assert exit_flag is False and reason == ""
    exit_flag, reason = m.on_tick(102.0, 2.0)
    assert exit_flag is True and reason == "tp"
    exit_flag, reason = m.on_tick(103.0, 3.0)
    assert exit_flag is True and reason == "tp"


def test_tbl_manager_sl_hit():
    """Price at or below entry-sl triggers exit with reason 'sl'."""
    m = TBLManager(entry_price=100.0, tp=2.0, sl=1.5, tmax=10.0, entry_time=0.0)
    exit_flag, reason = m.on_tick(98.5, 1.0)
    assert exit_flag is True and reason == "sl"
    m2 = TBLManager(100.0, 2.0, 1.5, 10.0, entry_time=0.0)
    exit_flag, reason = m2.on_tick(98.4, 0.5)
    assert exit_flag is True and reason == "sl"


def test_tbl_manager_time_hit():
    """t >= entry_time + tmax triggers exit with reason 'time'."""
    m = TBLManager(entry_price=100.0, tp=5.0, sl=5.0, tmax=3.0, entry_time=0.0)
    exit_flag, reason = m.on_tick(100.5, 2.9)
    assert exit_flag is False
    exit_flag, reason = m.on_tick(100.5, 3.0)
    assert exit_flag is True and reason == "time"


def test_tbl_manager_entry_time_first_tick():
    """If entry_time not given, first on_tick sets it."""
    m = TBLManager(entry_price=50.0, tp=10.0, sl=10.0, tmax=5.0)
    m.on_tick(50.0, 100.0)
    exit_flag, reason = m.on_tick(50.0, 104.9)
    assert exit_flag is False
    exit_flag, reason = m.on_tick(50.0, 105.0)
    assert exit_flag is True and reason == "time"


def test_tbl_manager_no_exit_inside_barriers():
    """Price strictly between entry-sl and entry+tp and t < entry+tmax: no exit."""
    m = TBLManager(entry_price=100.0, tp=3.0, sl=2.0, tmax=10.0, entry_time=0.0)
    for price in [99.0, 101.0, 102.5]:
        exit_flag, reason = m.on_tick(price, 5.0)
        assert exit_flag is False and reason == ""


def test_var_future_formula():
    """var_future(theta, sigma, T) = (σ²/(2θ)) * (1 - exp(-2θ T))."""
    theta, sigma, T = 0.5, 2.0, 1.0
    v = var_future(theta, sigma, T)
    expected = (sigma ** 2 / (2 * theta)) * (1.0 - np.exp(-2 * theta * T))
    assert abs(v - expected) < 1e-10
    assert v > 0


def test_var_future_zero_T():
    """T=0 => variance 0."""
    assert var_future(1.0, 1.0, 0.0) == 0.0


def test_estimate_ou_params_returns_three():
    """estimate_ou_params returns (theta, mu, sigma)."""
    np.random.seed(42)
    x = 100 + np.cumsum(np.random.randn(100) * 0.5)
    series = pd.Series(x)
    theta, mu, sigma = estimate_ou_params(series)
    assert isinstance(theta, float) and isinstance(mu, float) and isinstance(sigma, float)
    assert theta > 0
    assert sigma >= 0


def test_estimate_ou_params_insufficient_raises():
    """Too few points raises."""
    with pytest.raises(ValueError, match="at least 3"):
        estimate_ou_params(pd.Series([1.0, 2.0]))


def test_ou_params_on_synthetic_ou():
    """On synthetic OU series, estimate_ou_params returns plausible (theta, mu, sigma)."""
    np.random.seed(43)
    n = 200
    theta_true, mu_true, sigma_true = 0.2, 100.0, 1.0
    x = np.zeros(n)
    x[0] = mu_true
    for t in range(1, n):
        x[t] = x[t - 1] + theta_true * (mu_true - x[t - 1]) + sigma_true * np.random.randn()
    series = pd.Series(x)
    theta, mu, sigma = estimate_ou_params(series)
    assert theta > 0
    assert abs(mu - mu_true) < 20
    assert 0.1 < sigma < 5.0
