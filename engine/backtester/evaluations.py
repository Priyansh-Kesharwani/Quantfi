import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


def score_vs_forward_returns(
    scores: np.ndarray,
    prices: np.ndarray,
    forward_windows: List[int] = [5, 10, 20, 60],
    n_quantiles: int = 10
) -> Dict[str, Any]:
    n = len(scores)
    results = {
        "forward_windows": forward_windows,
        "n_quantiles": n_quantiles,
        "decile_analysis": {}
    }
    
    for window in forward_windows:
        fwd_returns = np.full(n, np.nan)
        for i in range(n - window):
            if prices[i] > 0:
                fwd_returns[i] = (prices[i + window] - prices[i]) / prices[i]
        
        valid_mask = ~np.isnan(scores) & ~np.isnan(fwd_returns)
        valid_scores = scores[valid_mask]
        valid_returns = fwd_returns[valid_mask]
        
        if len(valid_scores) < n_quantiles * 10:
            logger.warning(f"Insufficient data for {window}d analysis")
            continue
        
        quantile_bounds = np.percentile(valid_scores, np.linspace(0, 100, n_quantiles + 1))
        
        decile_stats = []
        for q in range(n_quantiles):
            low = quantile_bounds[q]
            high = quantile_bounds[q + 1]
            
            if q == n_quantiles - 1:
                mask = (valid_scores >= low) & (valid_scores <= high)
            else:
                mask = (valid_scores >= low) & (valid_scores < high)
            
            decile_returns = valid_returns[mask]
            
            if len(decile_returns) > 0:
                decile_stats.append({
                    "decile": q + 1,
                    "score_range": (float(low), float(high)),
                    "n_obs": len(decile_returns),
                    "mean_return": float(np.mean(decile_returns)),
                    "median_return": float(np.median(decile_returns)),
                    "std_return": float(np.std(decile_returns)),
                    "hit_rate": float(np.mean(decile_returns > 0)),              
                    "sharpe_like": float(np.mean(decile_returns) / (np.std(decile_returns) + 1e-10))
                })
        
        results["decile_analysis"][f"{window}d"] = decile_stats
        
        if decile_stats:
            means = [d["mean_return"] for d in decile_stats]
            ranks = np.arange(1, len(means) + 1)
            if len(means) > 2:
                corr = np.corrcoef(ranks, means)[0, 1]
                results["decile_analysis"][f"{window}d_monotonicity"] = float(corr)
    
    return results


def score_vs_forward_volatility(
    scores: np.ndarray,
    prices: np.ndarray,
    forward_windows: List[int] = [5, 10, 20],
    n_quantiles: int = 5
) -> Dict[str, Any]:
    n = len(scores)
    results = {
        "forward_windows": forward_windows,
        "n_quantiles": n_quantiles,
        "volatility_by_score": {}
    }
    
    safe_prices = np.maximum(prices, 1e-10)
    log_returns = np.diff(np.log(safe_prices))
    log_returns = np.insert(log_returns, 0, 0)
    
    for window in forward_windows:
        fwd_vol = np.full(n, np.nan)
        for i in range(n - window):
            fwd_rets = log_returns[i + 1:i + window + 1]
            valid_rets = fwd_rets[~np.isnan(fwd_rets)]
            if len(valid_rets) >= window // 2:
                fwd_vol[i] = np.std(valid_rets) * np.sqrt(252)
        
        valid_mask = ~np.isnan(scores) & ~np.isnan(fwd_vol)
        valid_scores = scores[valid_mask]
        valid_vol = fwd_vol[valid_mask]
        
        if len(valid_scores) < n_quantiles * 10:
            continue
        
        quantile_bounds = np.percentile(valid_scores, np.linspace(0, 100, n_quantiles + 1))
        
        bucket_stats = []
        for q in range(n_quantiles):
            low = quantile_bounds[q]
            high = quantile_bounds[q + 1]
            
            if q == n_quantiles - 1:
                mask = (valid_scores >= low) & (valid_scores <= high)
            else:
                mask = (valid_scores >= low) & (valid_scores < high)
            
            bucket_vol = valid_vol[mask]
            
            if len(bucket_vol) > 0:
                bucket_stats.append({
                    "bucket": q + 1,
                    "score_range": (float(low), float(high)),
                    "n_obs": len(bucket_vol),
                    "mean_vol": float(np.mean(bucket_vol)),
                    "median_vol": float(np.median(bucket_vol)),
                    "std_vol": float(np.std(bucket_vol))
                })
        
        results["volatility_by_score"][f"{window}d"] = bucket_stats
    
    return results


def dca_cost_comparison(
    scores: np.ndarray,
    prices: np.ndarray,
    high_score_threshold: float = 70.0,
    low_score_threshold: float = 30.0,
    investment_per_period: float = 1000.0,
    frequency: int = 5                    
) -> Dict[str, Any]:
    n = len(scores)
    results = {
        "high_score_threshold": high_score_threshold,
        "low_score_threshold": low_score_threshold,
        "investment_per_period": investment_per_period,
        "frequency": frequency,
        "strategies": {}
    }
    
    strategies = {
        "uniform": lambda s, p: investment_per_period,
        "high_score_boost": lambda s, p: investment_per_period * (1.5 if s >= high_score_threshold else 0.75),
        "low_score_boost": lambda s, p: investment_per_period * (1.5 if s <= low_score_threshold else 0.75),
        "proportional": lambda s, p: investment_per_period * (s / 50.0) if not np.isnan(s) else investment_per_period
    }
    
    for strategy_name, invest_func in strategies.items():
        total_invested = 0.0
        total_units = 0.0
        purchases = []
        
        for i in range(0, n, frequency):
            if np.isnan(scores[i]) or prices[i] <= 0:
                continue
            
            invest_amount = invest_func(scores[i], prices[i])
            units = invest_amount / prices[i]
            
            total_invested += invest_amount
            total_units += units
            
            purchases.append({
                "day": i,
                "score": float(scores[i]),
                "price": float(prices[i]),
                "invested": float(invest_amount),
                "units": float(units)
            })
        
        if total_units > 0:
            avg_cost = total_invested / total_units
            final_value = total_units * prices[-1] if not np.isnan(prices[-1]) else 0
            total_return = (final_value - total_invested) / total_invested if total_invested > 0 else 0
            
            results["strategies"][strategy_name] = {
                "total_invested": float(total_invested),
                "total_units": float(total_units),
                "avg_cost_basis": float(avg_cost),
                "final_value": float(final_value),
                "total_return_pct": float(total_return * 100),
                "n_purchases": len(purchases)
            }
    
    if "uniform" in results["strategies"] and "high_score_boost" in results["strategies"]:
        uniform_cost = results["strategies"]["uniform"]["avg_cost_basis"]
        boost_cost = results["strategies"]["high_score_boost"]["avg_cost_basis"]
        results["cost_improvement_pct"] = float((uniform_cost - boost_cost) / uniform_cost * 100)
    
    return results


def drawdown_analysis(
    scores: np.ndarray,
    prices: np.ndarray,
    drawdown_threshold: float = -0.10,        
    score_buckets: List[Tuple[float, float]] = [(0, 30), (30, 50), (50, 70), (70, 100)]
) -> Dict[str, Any]:
    n = len(scores)
    results = {
        "drawdown_threshold": drawdown_threshold,
        "score_buckets": score_buckets,
        "drawdown_stats": {}
    }
    
    rolling_max = np.maximum.accumulate(prices)
    drawdown = (prices - rolling_max) / np.maximum(rolling_max, 1e-10)
    
    in_drawdown = drawdown < drawdown_threshold
    
    dd_scores = scores[in_drawdown & ~np.isnan(scores)]
    normal_scores = scores[~in_drawdown & ~np.isnan(scores)]
    
    results["drawdown_stats"]["avg_score_during_dd"] = float(np.mean(dd_scores)) if len(dd_scores) > 0 else None
    results["drawdown_stats"]["avg_score_normal"] = float(np.mean(normal_scores)) if len(normal_scores) > 0 else None
    results["drawdown_stats"]["pct_time_in_dd"] = float(np.sum(in_drawdown) / n * 100)
    
    dd_events = []
    i = 0
    while i < n - 1:
        if not in_drawdown[i] and in_drawdown[i + 1]:
            start = i + 1
            j = start
            while j < n and in_drawdown[j]:
                j += 1
            end = j
            
            target_price = prices[start]
            recovery = end
            while recovery < n and prices[recovery] < target_price:
                recovery += 1
            
            if recovery < n:
                dd_events.append({
                    "start": start,
                    "end": end,
                    "recovery": recovery,
                    "max_dd": float(np.min(drawdown[start:end + 1])),
                    "duration_dd": end - start,
                    "duration_recovery": recovery - end
                })
            
            i = end
        else:
            i += 1
    
    results["n_drawdown_events"] = len(dd_events)
    
    if dd_events:
        entry_analysis = {bucket: [] for bucket in score_buckets}
        
        for event in dd_events:
            for day in range(event["start"], min(event["end"] + 1, n)):
                if np.isnan(scores[day]):
                    continue
                
                score = scores[day]
                for low, high in score_buckets:
                    if low <= score < high or (score == high and high == 100):
                        entry_analysis[(low, high)].append({
                            "entry_day": day,
                            "score": score,
                            "price": prices[day],
                            "recovery_day": event["recovery"],
                            "recovery_days": event["recovery"] - day
                        })
                        break
        
        results["entry_by_score_bucket"] = {}
        for bucket, entries in entry_analysis.items():
            if entries:
                results["entry_by_score_bucket"][f"{bucket[0]}-{bucket[1]}"] = {
                    "n_entries": len(entries),
                    "avg_recovery_days": float(np.mean([e["recovery_days"] for e in entries]))
                }
    
    return results


def crisis_regime_analysis(
    scores: np.ndarray,
    prices: np.ndarray,
    dates: Optional[np.ndarray] = None,
    crisis_periods: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    n = len(scores)
    results = {
        "n_observations": n,
        "regime_analysis": {}
    }
    
    safe_prices = np.maximum(prices, 1e-10)
    log_returns = np.diff(np.log(safe_prices))
    log_returns = np.insert(log_returns, 0, 0)
    
    vol_window = 21
    rolling_vol = np.full(n, np.nan)
    for i in range(vol_window, n):
        rolling_vol[i] = np.std(log_returns[i - vol_window:i]) * np.sqrt(252)
    
    valid_vol = rolling_vol[~np.isnan(rolling_vol)]
    if len(valid_vol) > 0:
        crisis_threshold = np.percentile(valid_vol, 80)
        is_crisis = rolling_vol >= crisis_threshold
    else:
        is_crisis = np.zeros(n, dtype=bool)
    
    crisis_scores = scores[is_crisis & ~np.isnan(scores)]
    normal_scores = scores[~is_crisis & ~np.isnan(scores)]
    
    results["regime_analysis"]["crisis"] = {
        "n_obs": len(crisis_scores),
        "mean_score": float(np.mean(crisis_scores)) if len(crisis_scores) > 0 else None,
        "median_score": float(np.median(crisis_scores)) if len(crisis_scores) > 0 else None,
        "std_score": float(np.std(crisis_scores)) if len(crisis_scores) > 0 else None,
        "pct_above_50": float(np.mean(crisis_scores > 50) * 100) if len(crisis_scores) > 0 else None
    }
    
    results["regime_analysis"]["normal"] = {
        "n_obs": len(normal_scores),
        "mean_score": float(np.mean(normal_scores)) if len(normal_scores) > 0 else None,
        "median_score": float(np.median(normal_scores)) if len(normal_scores) > 0 else None,
        "std_score": float(np.std(normal_scores)) if len(normal_scores) > 0 else None,
        "pct_above_50": float(np.mean(normal_scores > 50) * 100) if len(normal_scores) > 0 else None
    }
    
    if results["regime_analysis"]["crisis"]["mean_score"] is not None and\
       results["regime_analysis"]["normal"]["mean_score"] is not None:
        results["regime_analysis"]["score_diff"] = (
            results["regime_analysis"]["crisis"]["mean_score"] - 
            results["regime_analysis"]["normal"]["mean_score"]
        )
    
    if dates is not None and crisis_periods:
        dates = pd.to_datetime(dates)
        
        results["named_crises"] = {}
        for period in crisis_periods:
            name = period["name"]
            start = pd.to_datetime(period["start"])
            end = pd.to_datetime(period["end"])
            
            mask = (dates >= start) & (dates <= end)
            period_scores = scores[mask & ~np.isnan(scores)]
            period_returns = log_returns[mask]
            
            if len(period_scores) > 0:
                results["named_crises"][name] = {
                    "start": str(start),
                    "end": str(end),
                    "n_days": int(np.sum(mask)),
                    "mean_score": float(np.mean(period_scores)),
                    "min_score": float(np.min(period_scores)),
                    "max_score": float(np.max(period_scores)),
                    "cumulative_return": float(np.sum(period_returns))
                }
    
    return results
