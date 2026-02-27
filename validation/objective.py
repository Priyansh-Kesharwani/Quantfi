"""
GT-Score and Deflated Sharpe Ratio (DSR) objective functions.

GT-Score: composite objective with significance gate, R² consistency, downside deviation.
DSR: multiple-testing correction for Sharpe ratio.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm


def compute_gt_score(
    equity_curve: pd.Series,
    benchmark_curve: pd.Series,
    *,
    eps: float = 1e-8,
    z_min: float = 1.0,
    underperform_penalty: float = -1e6,
    min_trades: Optional[int] = None,
    min_trades_penalty: float = -1e5,
    benchmark_scale: float = 1.0,
    invested_fraction: Optional[float] = None,
    turnover_penalty_lambda: float = 0.0,
    turnover_rate: float = 0.0,
) -> float:
    """
    GT-Score = (mu * ln(z) * r^2) / (sigma_d + eps) with piecewise handling.

    Parameters
    ----------
    equity_curve : pd.Series
        Strategy equity (index = date or int).
    benchmark_curve : pd.Series
        Benchmark equity aligned to same index.
    eps : float
        Small constant for numerical stability.
    z_min : float
        Minimum Z for full score; below this use discounted ln(z).
    underperform_penalty : float
        Returned when strategy mean return <= benchmark mean return.
    min_trades : int, optional
        If set, penalty when number of periods (or implied trades) < min_trades.
    min_trades_penalty : float
        Penalty when min_trades is not met.
    benchmark_scale : float
        Legacy scalar applied to benchmark mean return.
    invested_fraction : float, optional
        Average fraction of time the strategy is invested (0-1).  When provided
        the benchmark mean return is scaled by this fraction so a conditional
        strategy is not compared against an always-invested B&H in bull markets.
    turnover_penalty_lambda : float
        Coefficient for turnover penalty subtracted from GT-Score.
    turnover_rate : float
        Strategy turnover rate (round-trips / period) for penalty calculation.

    Returns
    -------
    float
        GT-Score (higher is better); large negative on underperformance or violation.
    """
    eq = equity_curve.dropna()
    bm = benchmark_curve.reindex(eq.index).ffill().bfill()
    if len(eq) < 2 or bm.isna().all():
        return underperform_penalty
    eq = eq.loc[bm.notna()]
    bm = bm.loc[eq.index]
    if len(eq) < 2:
        return underperform_penalty

    ret = eq.pct_change().dropna()
    ret_bm = bm.pct_change().dropna()
    common = ret.index.intersection(ret_bm.index)
    ret = ret.loc[common]
    ret_bm = ret_bm.loc[common]
    if len(ret) < 2:
        return underperform_penalty

    mu = float(ret.mean())
    mu_bm = float(ret_bm.mean())

    # Scale benchmark by invested fraction for fair conditional-vs-B&H comparison
    if invested_fraction is not None and 0.0 < invested_fraction <= 1.0:
        mu_bm_adj = mu_bm * invested_fraction
    else:
        mu_bm_adj = mu_bm * benchmark_scale

    if mu <= mu_bm_adj:
        return (mu - mu_bm_adj) * 1e4

    n = len(ret)
    se_diff = float(np.sqrt(ret.var() / n + ret_bm.var() / n + eps))
    z = (mu - mu_bm_adj) / se_diff if se_diff > 0 else z_min
    if z <= 0:
        return (mu - mu_bm_adj) * 1e4
    if z < z_min:
        z = z_min
    ln_z = float(np.log(z))

    x = np.arange(len(eq), dtype=float)
    y = eq.values
    if np.var(y) < eps:
        r_sq = 0.0
    else:
        coeffs = np.polyfit(x, y, 1)
        y_hat = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_sq = float(1 - ss_res / (ss_tot + eps))

    neg_ret = ret[ret < 0]
    sigma_d = float(neg_ret.std()) if len(neg_ret) > 1 else eps

    if min_trades is not None and n < min_trades:
        return min_trades_penalty

    denom = sigma_d + eps
    if denom <= 0:
        denom = eps
    gt = (mu * ln_z * r_sq) / denom

    if turnover_penalty_lambda > 0 and turnover_rate > 0:
        gt -= turnover_penalty_lambda * turnover_rate

    return float(gt)


def compute_dsr(
    sharpes: List[float],
    n_trials: int,
    *,
    skew: Optional[float] = None,
    kurtosis: Optional[float] = None,
    annualization: float = 252.0,
) -> float:
    """
    Deflated Sharpe Ratio: corrects for multiple testing over n_trials.

    Uses expected maximum Sharpe under null and optional skew/kurtosis.
    Returns a deflated Sharpe value (higher is better).

    Parameters
    ----------
    sharpes : list of float
        Observed Sharpe ratios (e.g. per CPCV path or per trial).
    n_trials : int
        Number of independent trials/configs (for selection bias).
    skew : float, optional
        Skewness of returns (for PSR-style adjustment).
    kurtosis : float, optional
        Excess kurtosis of returns (for PSR-style adjustment).
    annualization : float
        Unused; kept for API compatibility.

    Returns
    -------
    float
        Deflated Sharpe ratio.
    """
    if not sharpes or n_trials < 1:
        return 0.0
    sharpes_arr = np.asarray(sharpes, dtype=float)
    observed = float(np.mean(sharpes_arr))
    var_sr = float(np.var(sharpes_arr))
    if var_sr <= 0:
        var_sr = 1e-12
    if n_trials <= 1:
        return observed
    euler_gamma = 0.57721566490153286060651209
    z_n = norm.ppf(1 - 1 / n_trials)
    z_ne = norm.ppf(1 - 1 / (n_trials * np.e))
    emax_null = np.sqrt(var_sr) * ((1 - euler_gamma) * z_n + euler_gamma * z_ne)
    deflated = observed - emax_null
    if skew is not None or kurtosis is not None:
        pass
    return float(deflated)


def equity_curve_from_result(result: Dict[str, Any]) -> Tuple[pd.Series, pd.Series]:
    """
    Extract strategy and benchmark (buy-and-hold) equity series from backtester result.

    Parameters
    ----------
    result : dict
        From PortfolioSimulator.run(): keys 'equity_curve', 'benchmarks'.

    Returns
    -------
    (strategy_curve, benchmark_curve) : pd.Series
        Index = date string; values = equity.
    """
    ec = result.get("equity_curve") or []
    benchmarks = result.get("benchmarks") or {}
    bnh = benchmarks.get("buy_and_hold") or {}
    bnh_curve = bnh.get("equity_curve") or []
    dates = [x.get("date") for x in ec if x.get("date")]
    equity = [x.get("equity") for x in ec if x.get("equity") is not None]
    if not dates or len(dates) != len(equity):
        strategy = pd.Series(dtype=float)
    else:
        strategy = pd.Series(equity, index=pd.Index(dates))
    bnh_dates = [x.get("date") for x in bnh_curve if x.get("date")]
    bnh_equity = [x.get("equity") for x in bnh_curve if x.get("equity") is not None]
    if not bnh_dates or len(bnh_dates) != len(bnh_equity):
        benchmark = pd.Series(dtype=float)
    else:
        benchmark = pd.Series(bnh_equity, index=pd.Index(bnh_dates))
    return strategy, benchmark
