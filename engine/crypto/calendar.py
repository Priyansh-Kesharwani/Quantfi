"""Crypto calendar utilities: annualization factors, performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

ANNUALIZATION_FACTORS = {
    "1m": 525_600,
    "5m": 105_120,
    "15m": 35_040,
    "1h": 8_760,
    "4h": 2_190,
    "1d": 365,
}

TIMEFRAME_TO_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

def get_annualization_factor(timeframe: str) -> int:
    """Return the number of bars per year for a given timeframe."""
    if timeframe not in ANNUALIZATION_FACTORS:
        raise ValueError(
            f"Unknown timeframe '{timeframe}'. Valid: {list(ANNUALIZATION_FACTORS.keys())}"
        )
    return ANNUALIZATION_FACTORS[timeframe]

def bars_per_day(timeframe: str) -> float:
    """Number of bars in a calendar day."""
    return get_annualization_factor(timeframe) / 365.0

_SHARPE_CAP = 50.0
_CAGR_CAP = 99.99
_CALMAR_CAP = 200.0
_MIN_YEARS_FOR_ANNUALIZATION = 0.08

def _n_years(n_bars: int, timeframe: str) -> float:
    ann = get_annualization_factor(timeframe)
    return n_bars / ann if ann > 0 else 0.0

def annualized_sharpe(
    returns: pd.Series,
    timeframe: str,
    risk_free_annual: float = 0.0,
) -> float:
    """Annualized Sharpe ratio from bar-level returns, capped to ±50."""
    if len(returns) < 2 or returns.std() <= 0:
        return 0.0
    ann = get_annualization_factor(timeframe)
    rf_per_bar = (1.0 + risk_free_annual) ** (1.0 / ann) - 1.0
    excess = returns - rf_per_bar
    raw = float((excess.mean() / excess.std()) * np.sqrt(ann))
    return float(np.clip(raw, -_SHARPE_CAP, _SHARPE_CAP))

def annualized_sortino(
    returns: pd.Series,
    timeframe: str,
    risk_free_annual: float = 0.0,
) -> float:
    """Annualized Sortino ratio (downside deviation only), capped to ±50."""
    if len(returns) < 2:
        return 0.0
    ann = get_annualization_factor(timeframe)
    rf_per_bar = (1.0 + risk_free_annual) ** (1.0 / ann) - 1.0
    excess = returns - rf_per_bar
    downside = excess[excess < 0]
    if len(downside) < 1 or downside.std() <= 0:
        return _SHARPE_CAP if excess.mean() > 0 else 0.0
    raw = float((excess.mean() / downside.std()) * np.sqrt(ann))
    return float(np.clip(raw, -_SHARPE_CAP, _SHARPE_CAP))

def annualized_cagr(equity_curve: pd.Series, timeframe: str) -> float:
    """CAGR from an equity curve.

    For sub-month backtests, returns simple total return instead of
    extrapolating to a misleading annual figure.
    """
    if len(equity_curve) < 2:
        return 0.0
    initial = equity_curve.iloc[0]
    final = equity_curve.iloc[-1]
    if initial <= 0 or final <= 0:
        return 0.0
    n_bars = len(equity_curve)
    ny = _n_years(n_bars, timeframe)
    if ny <= 0:
        return 0.0
    if ny < _MIN_YEARS_FOR_ANNUALIZATION:
        return float(final / initial - 1.0)
    raw = float((final / initial) ** (1.0 / ny) - 1.0)
    return float(np.clip(raw, -_CAGR_CAP, _CAGR_CAP))

def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum drawdown as a negative fraction (e.g. -0.15 = 15% drawdown)."""
    if len(equity_curve) < 2:
        return 0.0
    peak = equity_curve.expanding().max()
    dd = (equity_curve - peak) / (peak + 1e-12)
    return float(dd.min())

def calmar_ratio(equity_curve: pd.Series, timeframe: str) -> float:
    """Calmar ratio = CAGR / |max drawdown|, capped to ±200."""
    cagr = annualized_cagr(equity_curve, timeframe)
    mdd = abs(max_drawdown(equity_curve))
    if mdd < 1e-8:
        return min(cagr * 100.0, _CALMAR_CAP) if cagr > 0 else 0.0
    raw = cagr / mdd
    return float(np.clip(raw, -_CALMAR_CAP, _CALMAR_CAP))

def profit_factor(trades_pnl: pd.Series) -> float:
    """Gross profit / gross loss."""
    gains = trades_pnl[trades_pnl > 0].sum()
    losses = abs(trades_pnl[trades_pnl < 0].sum())
    if losses < 1e-12:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)

def bootstrap_sharpe_pvalue(
    returns: pd.Series,
    timeframe: str,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict:
    """Bootstrap test for annualized Sharpe significance.

    Shuffles return order *n_bootstrap* times, recomputes Sharpe each time,
    and returns the p-value (fraction of shuffled Sharpes >= observed).
    """
    observed = annualized_sharpe(returns, timeframe)
    rng = np.random.RandomState(seed)
    arr = returns.values.copy()
    count_ge = 0
    for _ in range(n_bootstrap):
        rng.shuffle(arr)
        shuffled_sharpe = annualized_sharpe(pd.Series(arr), timeframe)
        if shuffled_sharpe >= observed:
            count_ge += 1
    p_value = count_ge / n_bootstrap
    return {
        "observed_sharpe": observed,
        "p_value": p_value,
        "n_bootstrap": n_bootstrap,
        "significant_5pct": p_value < 0.05,
    }
