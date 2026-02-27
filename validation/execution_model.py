"""
Phase 3 — Execution Risk & Slippage Model.

Models fill-price slippage, market-impact, and transaction costs
for realistic backtest P&L estimation.

Impact model (Kyle/Kelly style):
    fill_price = mid + sign * k_impact * (order_size / ADV)^gamma + N(0, sigma)

Also provides latency jitter for intraday simulations and a combined
execution-cost calculator for walk-forward / tuning backtests.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class ExecutionConfig:
    """Configuration for the execution-cost model."""
    k_impact: float = 0.001
    gamma: float = 0.5
    sigma_slip: float = 0.0001
    order_size_pct: float = 0.01
    commission_bps: float = 2.0
    spread_bps: float = 1.0
    mean_latency_ms: float = 50.0
    latency_jitter: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExecutionConfig":
        slip = d.get("slippage", {})
        lat = d.get("latency", {})
        tc = d.get("transaction_costs", {})
        return cls(
            k_impact=slip.get("k_impact", 0.001),
            gamma=slip.get("gamma", 0.5),
            sigma_slip=slip.get("sigma_slip", 0.0001),
            order_size_pct=slip.get("order_size_pct", 0.01),
            commission_bps=tc.get("commission_bps", 2.0),
            spread_bps=tc.get("spread_bps", 1.0),
            mean_latency_ms=lat.get("mean_latency_ms", 50.0),
            latency_jitter=lat.get("jitter", True),
        )

def market_impact(
    order_size: float,
    adv: float,
    k_impact: float = 0.001,
    gamma: float = 0.5,
) -> float:
    """Compute price impact using Kyle/Kelly model.

    impact = k * (Q / ADV)^γ

    Parameters
    ----------
    order_size : float
        Notional order size (shares or dollars).
    adv : float
        Average daily volume (same units as order_size).
    k_impact : float
        Impact coefficient.
    gamma : float
        Impact exponent (typically 0.5 for sqrt-law).

    Returns
    -------
    float
        Price impact as a fraction of mid-price.
    """
    if adv <= 0:
        return 0.0
    ratio = abs(order_size) / adv
    return k_impact * (ratio ** gamma)

def compute_fill_price(
    mid_price: float,
    side: int,
    order_size: float,
    adv: float,
    config: Optional[ExecutionConfig] = None,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[float, Dict[str, float]]:
    """Compute realistic fill price with impact + noise.

    fill_price = mid + sign * impact(Q, ADV) * mid + N(0, σ) * mid

    Parameters
    ----------
    mid_price : float
        Current mid-price.
    side : int
        +1 for buy, -1 for sell.
    order_size : float
        Order size (shares).
    adv : float
        Average daily volume.
    config : ExecutionConfig or None
        Execution parameters.
    rng : RandomState or None
        RNG for noise.

    Returns
    -------
    fill_price : float
    breakdown : dict
        Detailed cost breakdown.
    """
    if config is None:
        config = ExecutionConfig()
    if rng is None:
        rng = np.random.RandomState(0)

    impact_frac = market_impact(order_size, adv, config.k_impact, config.gamma)
    noise_frac = rng.normal(0, config.sigma_slip) if config.sigma_slip > 0 else 0.0
    spread_frac = config.spread_bps * 1e-4

    total_slip_frac = side * (impact_frac + spread_frac) + noise_frac
    fill = mid_price * (1.0 + total_slip_frac)

    commission_frac = config.commission_bps * 1e-4

    breakdown = {
        "mid_price": mid_price,
        "fill_price": fill,
        "impact_frac": impact_frac,
        "noise_frac": noise_frac,
        "spread_frac": spread_frac,
        "commission_frac": commission_frac,
        "total_slip_frac": abs(total_slip_frac),
        "total_cost_frac": abs(total_slip_frac) + commission_frac,
    }
    return fill, breakdown

def simulate_latency(
    n: int,
    mean_ms: float = 50.0,
    jitter: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """Simulate latency jitter for intraday execution.

    Parameters
    ----------
    n : int
        Number of latency samples.
    mean_ms : float
        Mean latency in milliseconds.
    jitter : bool
        If True, use exponential distribution; else constant.
    seed : int
        RNG seed.

    Returns
    -------
    np.ndarray
        Latency samples in milliseconds.
    """
    rng = np.random.RandomState(seed)
    if jitter and mean_ms > 0:
        return rng.exponential(mean_ms, size=n)
    return np.full(n, mean_ms)

def apply_execution_costs(
    returns: pd.Series,
    entry_signals: pd.Series,
    exit_signals: pd.Series,
    volumes: pd.Series,
    config: Optional[ExecutionConfig] = None,
    entry_threshold: float = 70.0,
    exit_threshold: float = 70.0,
    seed: int = 0,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Apply realistic execution costs to a signal-based return stream.

    For each trade (entry when score > threshold, exit when exit score > threshold),
    deduct slippage + commission from the trade returns.

    Parameters
    ----------
    returns : pd.Series
        Raw bar-to-bar returns.
    entry_signals : pd.Series
        Entry_Score values.
    exit_signals : pd.Series
        Exit_Score values.
    volumes : pd.Series
        Volume series for ADV calculation.
    config : ExecutionConfig
        Execution parameters.
    entry_threshold, exit_threshold : float
        Score thresholds.
    seed : int
        RNG seed.

    Returns
    -------
    adjusted_returns : pd.Series
        Returns after execution costs.
    cost_report : dict
        Aggregate cost statistics.
    """
    if config is None:
        config = ExecutionConfig()

    rng = np.random.RandomState(seed)
    n = len(returns)
    adj = returns.copy().values.astype(float)

    adv = volumes.rolling(20, min_periods=1).mean().values

    entry_vals = entry_signals.values
    exit_vals = exit_signals.values

    in_trade = False
    n_trades = 0
    total_entry_cost = 0.0
    total_exit_cost = 0.0

    for i in range(n):
        if not in_trade:
            if not np.isnan(entry_vals[i]) and entry_vals[i] > entry_threshold:
                in_trade = True
                n_trades += 1
                order_sz = config.order_size_pct * adv[i] if adv[i] > 0 else 1.0
                impact = market_impact(order_sz, adv[i], config.k_impact, config.gamma)
                spread = config.spread_bps * 1e-4
                comm = config.commission_bps * 1e-4
                noise = rng.normal(0, config.sigma_slip)
                cost = impact + spread + comm + abs(noise)
                adj[i] -= cost
                total_entry_cost += cost
        else:
            should_exit = (
                (not np.isnan(exit_vals[i]) and exit_vals[i] > exit_threshold)
                or i == n - 1
            )
            if should_exit:
                in_trade = False
                order_sz = config.order_size_pct * adv[i] if adv[i] > 0 else 1.0
                impact = market_impact(order_sz, adv[i], config.k_impact, config.gamma)
                spread = config.spread_bps * 1e-4
                comm = config.commission_bps * 1e-4
                noise = rng.normal(0, config.sigma_slip)
                cost = impact + spread + comm + abs(noise)
                adj[i] -= cost
                total_exit_cost += cost

    adjusted = pd.Series(adj, index=returns.index, name="adj_returns")

    cost_report = {
        "n_trades": n_trades,
        "total_entry_cost_frac": total_entry_cost,
        "total_exit_cost_frac": total_exit_cost,
        "avg_cost_per_trade": (total_entry_cost + total_exit_cost) / max(n_trades, 1),
        "pnl_erosion": float(returns.sum() - adjusted.sum()),
    }

    return adjusted, cost_report

def slippage_sensitivity_matrix(
    returns: pd.Series,
    entry_signals: pd.Series,
    exit_signals: pd.Series,
    volumes: pd.Series,
    k_impact_range: Optional[List[float]] = None,
    gamma_range: Optional[List[float]] = None,
    entry_threshold: float = 70.0,
    exit_threshold: float = 70.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Compute slippage sensitivity matrix over impact parameters.

    Parameters
    ----------
    returns, entry_signals, exit_signals, volumes :
        Standard signal and market data.
    k_impact_range : list of float
        Range of k_impact values to test.
    gamma_range : list of float
        Range of gamma values to test.

    Returns
    -------
    pd.DataFrame
        Index = k_impact, columns = gamma, values = total PnL erosion.
    """
    if k_impact_range is None:
        k_impact_range = [0.0, 0.0005, 0.001, 0.002, 0.005]
    if gamma_range is None:
        gamma_range = [0.3, 0.5, 0.7, 1.0]

    results = {}
    for k in k_impact_range:
        row = {}
        for g in gamma_range:
            cfg = ExecutionConfig(k_impact=k, gamma=g)
            _, report = apply_execution_costs(
                returns, entry_signals, exit_signals, volumes,
                config=cfg,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
                seed=seed,
            )
            row[g] = report["pnl_erosion"]
        results[k] = row

    return pd.DataFrame(results).T
