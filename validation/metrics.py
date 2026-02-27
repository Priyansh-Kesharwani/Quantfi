"""
Phase B — Validation Metrics Module.

Provides two categories of metrics:

1. **Score Metrics**: Evaluate the predictive quality of Entry_Score / Exit_Score
   against forward returns (IC, hit rate, Sortino, CAGR, max drawdown).

2. **Signal Metrics**: Evaluate trade-level quality once signals are converted
   to discrete entries/exits (holding period, latency, ROI, win/loss, profit factor).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from scipy.stats import spearmanr
import logging

logger = logging.getLogger(__name__)

def information_coefficient(
    scores: pd.Series,
    forward_returns: pd.Series,
) -> float:
    """Spearman rank correlation between scores and forward returns.

    Parameters
    ----------
    scores : pd.Series
        Entry_Score or Exit_Score (0-100).
    forward_returns : pd.Series
        Forward returns aligned to same index.

    Returns
    -------
    float
        Spearman correlation coefficient.
    """
    mask = scores.notna() & forward_returns.notna()
    s = scores[mask].values
    r = forward_returns[mask].values
    if len(s) < 3:
        return np.nan
    corr, _p = spearmanr(s, r)
    return float(corr)

def hit_rate(
    scores: pd.Series,
    forward_returns: pd.Series,
    threshold: float = 70.0,
) -> float:
    """Fraction of times score > threshold leads to positive forward return.

    Parameters
    ----------
    scores : pd.Series
        Entry_Score values.
    forward_returns : pd.Series
        Forward returns aligned to same index.
    threshold : float
        Score threshold (default 70) above which signal is "active".

    Returns
    -------
    float
        Hit rate ∈ [0, 1], or NaN if no signals above threshold.
    """
    mask = scores.notna() & forward_returns.notna() & (scores > threshold)
    if mask.sum() == 0:
        return np.nan
    return float((forward_returns[mask] > 0).mean())

def sortino_ratio(
    returns: pd.Series,
    target: float = 0.0,
    annualization: float = 252.0,
) -> float:
    """Sortino ratio: expected excess return / downside deviation.

    Parameters
    ----------
    returns : pd.Series
        Period returns.
    target : float
        Minimum acceptable return (default 0).
    annualization : float
        Annualisation factor (252 for daily).

    Returns
    -------
    float
        Annualised Sortino ratio.
    """
    excess = returns - target
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return np.inf if excess.mean() > 0 else 0.0
    downside_std = downside.std()
    return float(excess.mean() / downside_std * np.sqrt(annualization))

def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum peak-to-trough drawdown.

    Parameters
    ----------
    equity_curve : pd.Series
        Cumulative equity / wealth series.

    Returns
    -------
    float
        Maximum drawdown as a negative fraction (e.g. -0.15 = -15%).
    """
    peak = equity_curve.expanding().max()
    dd = (equity_curve - peak) / peak
    return float(dd.min())

def cagr(
    equity_curve: pd.Series,
    periods_per_year: float = 252.0,
) -> float:
    """Compound Annual Growth Rate.

    Parameters
    ----------
    equity_curve : pd.Series
        Cumulative equity series (starting value assumed > 0).
    periods_per_year : float
        Trading periods per year (252 for daily).

    Returns
    -------
    float
        CAGR as a decimal (e.g. 0.12 = 12%).
    """
    valid = equity_curve.dropna()
    if len(valid) < 2 or valid.iloc[0] <= 0:
        return 0.0
    total_return = valid.iloc[-1] / valid.iloc[0]
    n_periods = len(valid) - 1
    years = n_periods / periods_per_year
    if years <= 0:
        return 0.0
    return float(total_return ** (1.0 / years) - 1.0)

def forward_returns(
    prices: pd.Series,
    horizon: int = 5,
) -> pd.Series:
    """Compute forward returns over *horizon* bars.

    Parameters
    ----------
    prices : pd.Series
        Close price series.
    horizon : int
        Number of bars to look forward.

    Returns
    -------
    pd.Series
        Forward returns (NaN for last *horizon* bars).
    """
    fwd = prices.shift(-horizon) / prices - 1.0
    fwd.name = f"fwd_ret_{horizon}"
    return fwd

def evaluate_signals(
    entry: pd.Series,
    exit_: pd.Series,
    returns: pd.Series,
    entry_threshold: float = 70.0,
    exit_threshold: float = 70.0,
) -> Dict[str, Any]:
    """Evaluate trade-level signal quality.

    Converts continuous Entry/Exit scores into discrete trades using
    threshold-based logic and computes per-trade metrics.

    Parameters
    ----------
    entry : pd.Series
        Entry_Score ∈ [0, 100].
    exit_ : pd.Series
        Exit_Score ∈ [0, 100].
    returns : pd.Series
        Period (bar-to-bar) returns aligned to same index.
    entry_threshold : float
        Score above which an entry signal fires.
    exit_threshold : float
        Score above which an exit signal fires.

    Returns
    -------
    dict
        Keys: avg_holding_period, entry_latency, exit_lead,
              roi_per_trade, win_loss_ratio, profit_factor,
              n_trades, total_return.
    """
    common_idx = entry.index.intersection(exit_.index).intersection(returns.index)
    entry = entry.reindex(common_idx)
    exit_ = exit_.reindex(common_idx)
    returns = returns.reindex(common_idx)

    trades = _extract_trades(entry, exit_, returns, entry_threshold, exit_threshold)

    if len(trades) == 0:
        return {
            "avg_holding_period": np.nan,
            "entry_latency": np.nan,
            "exit_lead": np.nan,
            "roi_per_trade": np.nan,
            "win_loss_ratio": np.nan,
            "profit_factor": np.nan,
            "n_trades": 0,
            "total_return": 0.0,
        }

    holding_periods = [t["bars"] for t in trades]
    rois = [t["roi"] for t in trades]
    entry_latencies = [t["entry_latency"] for t in trades]
    exit_leads = [t["exit_lead"] for t in trades]

    wins = [r for r in rois if r > 0]
    losses = [r for r in rois if r <= 0]
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0

    return {
        "avg_holding_period": float(np.mean(holding_periods)),
        "entry_latency": float(np.nanmean(entry_latencies)),
        "exit_lead": float(np.nanmean(exit_leads)),
        "roi_per_trade": float(np.mean(rois)),
        "win_loss_ratio": float(len(wins) / max(len(losses), 1)),
        "profit_factor": float(gross_profit / max(gross_loss, 1e-12)),
        "n_trades": len(trades),
        "total_return": float(sum(rois)),
    }

def _extract_trades(
    entry: pd.Series,
    exit_: pd.Series,
    returns: pd.Series,
    entry_threshold: float,
    exit_threshold: float,
) -> List[Dict[str, Any]]:
    """Extract discrete trade segments from continuous scores.

    Logic:
    - Enter when Entry_Score > entry_threshold (and not already in trade)
    - Exit when Exit_Score > exit_threshold (and currently in trade)
    - Collect per-trade ROI, holding period, and timing metrics
    """
    trades = []
    in_trade = False
    entry_bar = 0
    cum_return = 0.0

    entry_vals = entry.values
    exit_vals = exit_.values
    ret_vals = returns.values
    cum_rets = (1 + returns).cumprod().values

    for i in range(len(entry)):
        if not in_trade:
            if not np.isnan(entry_vals[i]) and entry_vals[i] > entry_threshold:
                in_trade = True
                entry_bar = i
                cum_return = 0.0
        else:
            cum_return += ret_vals[i] if not np.isnan(ret_vals[i]) else 0.0

            should_exit = (
                (not np.isnan(exit_vals[i]) and exit_vals[i] > exit_threshold)
                or i == len(entry) - 1
            )

            if should_exit:
                bars = i - entry_bar
                if bars < 1:
                    bars = 1

                if entry_bar < len(cum_rets) and i < len(cum_rets):
                    trade_roi = cum_rets[i] / max(cum_rets[entry_bar], 1e-12) - 1.0
                else:
                    trade_roi = cum_return

                segment_rets = ret_vals[entry_bar:i + 1]
                segment_cum = np.nancumsum(segment_rets)
                if len(segment_cum) > 0:
                    entry_latency = float(np.argmax(segment_cum))
                else:
                    entry_latency = np.nan

                exit_lead = float(bars - np.argmax(segment_cum)) if len(segment_cum) > 0 else np.nan

                trades.append({
                    "entry_idx": entry_bar,
                    "exit_idx": i,
                    "bars": bars,
                    "roi": float(trade_roi),
                    "entry_latency": entry_latency,
                    "exit_lead": exit_lead,
                })
                in_trade = False

    return trades

def compute_score_metrics(
    scores: pd.Series,
    prices: pd.Series,
    forward_horizons: List[int] = None,
    entry_threshold: float = 70.0,
) -> Dict[str, Any]:
    """Compute a full suite of score-level metrics.

    Parameters
    ----------
    scores : pd.Series
        Entry_Score or Exit_Score.
    prices : pd.Series
        Close price series (same index alignment).
    forward_horizons : list of int
        Forward-return horizons to evaluate (default [5, 10, 20]).
    entry_threshold : float
        Score threshold for hit rate calculation.

    Returns
    -------
    dict
        Nested dict keyed by metric name.
    """
    if forward_horizons is None:
        forward_horizons = [5, 10, 20]

    results: Dict[str, Any] = {}

    for h in forward_horizons:
        fwd_ret = forward_returns(prices, horizon=h)
        results[f"ic_{h}d"] = information_coefficient(scores, fwd_ret)
        results[f"hit_rate_{h}d"] = hit_rate(scores, fwd_ret, threshold=entry_threshold)

    period_ret = prices.pct_change().fillna(0)

    mask = scores > entry_threshold
    signal_returns = period_ret.copy()
    signal_returns[~mask] = 0.0

    equity = (1 + signal_returns).cumprod()

    results["sortino"] = sortino_ratio(signal_returns)
    results["max_drawdown"] = max_drawdown(equity)
    results["cagr"] = cagr(equity)
    results["n_signals"] = int(mask.sum())
    results["pct_active"] = float(mask.mean())

    return results

def compute_all_metrics(
    entry_scores: pd.Series,
    exit_scores: pd.Series,
    prices: pd.Series,
    forward_horizons: List[int] = None,
    entry_threshold: float = 70.0,
    exit_threshold: float = 70.0,
) -> Dict[str, Any]:
    """Compute full metrics suite for both entry and exit scores.

    Returns
    -------
    dict with keys "entry_metrics", "exit_metrics", "signal_metrics".
    """
    if forward_horizons is None:
        forward_horizons = [5, 10, 20]

    period_ret = prices.pct_change().fillna(0)

    return {
        "entry_metrics": compute_score_metrics(
            entry_scores, prices, forward_horizons, entry_threshold
        ),
        "exit_metrics": compute_score_metrics(
            exit_scores, prices, forward_horizons, exit_threshold
        ),
        "signal_metrics": evaluate_signals(
            entry_scores, exit_scores, period_ret,
            entry_threshold, exit_threshold
        ),
    }
