import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

OOS_SORTINO_ANNUALIZATION = 252.0


def _dca_equity_curve_oos(
    scores: np.ndarray,
    prices: np.ndarray,
    threshold: float,
    multiplier: float,
    base_investment: float,
    frequency: int,
) -> np.ndarray:
    n = len(scores)
    equity = np.zeros(n)
    units = 0.0
    for t in range(n):
        if t % frequency == 0 and t < len(scores) and not np.isnan(scores[t]) and prices[t] > 0:
            invest = base_investment * multiplier if scores[t] >= threshold else base_investment
            units += invest / prices[t]
        equity[t] = units * prices[t] if prices[t] > 0 else (equity[t - 1] if t > 0 else 0.0)
    equity = np.maximum(equity, 1e-12)
    return equity


def _sortino_from_equity(equity: np.ndarray, annualization: float = OOS_SORTINO_ANNUALIZATION) -> float:
    if len(equity) < 2:
        return 0.0
    ret = np.diff(equity) / (equity[:-1] + 1e-12)
    downside = ret[ret < 0]
    if len(downside) == 0 or np.std(downside) <= 0:
        return np.inf if np.mean(ret) > 0 else 0.0
    return float(np.mean(ret) / np.std(downside) * np.sqrt(annualization))


@dataclass
class SweepConfig:
    thresholds: np.ndarray = None
    multipliers: np.ndarray = None
    base_investment: float = 1000.0
    frequency: int = 5
    forward_windows: List[int] = None
    oos_start_idx: Optional[int] = None

    def __post_init__(self):
        if self.thresholds is None:
            self.thresholds = np.arange(40, 80, 5.0)
        if self.multipliers is None:
            self.multipliers = np.array([1.0, 1.25, 1.5, 2.0])
        if self.forward_windows is None:
            self.forward_windows = [5, 10, 20]


def sweep_dca_thresholds(
    scores: np.ndarray,
    prices: np.ndarray,
    config: Optional[SweepConfig] = None,
) -> pd.DataFrame:
    if config is None:
        config = SweepConfig()

    start = config.oos_start_idx if config.oos_start_idx is not None else 0
    s = scores[start:]
    p = prices[start:]
    n = len(s)

    thresholds = config.thresholds
    multipliers = config.multipliers

    buy_days = np.arange(0, n, config.frequency)
    valid_mask = ~np.isnan(s[buy_days]) & (p[buy_days] > 0)
    buy_days = buy_days[valid_mask]

    if len(buy_days) == 0:
        logger.warning("No valid buy days found")
        return pd.DataFrame()

    buy_scores = s[buy_days]
    buy_prices = p[buy_days]
    above_thresh = buy_scores[:, None] >= thresholds[None, :]

    results = []
    for mult in multipliers:
        invest_matrix = np.where(
            above_thresh,
            config.base_investment * mult,
            config.base_investment,
        )
        units_matrix = invest_matrix / buy_prices[:, None]
        total_invested = invest_matrix.sum(axis=0)
        total_units = units_matrix.sum(axis=0)
        avg_cost = np.where(total_units > 0, total_invested / total_units, 0)
        final_value = total_units * p[-1]
        return_pct = np.where(
            total_invested > 0,
            (final_value - total_invested) / total_invested * 100,
            0,
        )

        for j, thresh in enumerate(thresholds):
            row = {
                "threshold": float(thresh),
                "multiplier": float(mult),
                "total_invested": float(total_invested[j]),
                "total_units": float(total_units[j]),
                "avg_cost": float(avg_cost[j]),
                "final_value": float(final_value[j]),
                "return_pct": float(return_pct[j]),
                "n_boosted_buys": int(above_thresh[:, j].sum()),
                "n_total_buys": len(buy_days),
            }
            eq = _dca_equity_curve_oos(
                s, p, float(thresh), float(mult),
                config.base_investment, config.frequency,
            )
            row["sortino_oos"] = float(_sortino_from_equity(eq))
            results.append(row)

    return pd.DataFrame(results)


def sweep_cadence_and_threshold(
    scores: np.ndarray,
    prices: np.ndarray,
    cadences: List[int] = None,
    thresholds: np.ndarray = None,
    base_investment: float = 1000.0,
    boost_mult: float = 1.5
) -> pd.DataFrame:
    if cadences is None:
        cadences = [1, 5, 10, 21]
    if thresholds is None:
        thresholds = np.arange(30, 80, 5.0)

    all_results = []

    for cadence in cadences:
        config = SweepConfig(
            thresholds=thresholds,
            multipliers=np.array([boost_mult]),
            base_investment=base_investment,
            frequency=cadence
        )
        df = sweep_dca_thresholds(scores, prices, config)
        if not df.empty:
            df['cadence_days'] = cadence
            all_results.append(df)

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


def compute_sweep_heatmap(
    sweep_results: pd.DataFrame,
    metric: str = "sortino_oos",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sweep_results.empty:
        return np.array([]), np.array([]), np.array([[]])

    thresholds = np.sort(sweep_results['threshold'].unique())
    multipliers = np.sort(sweep_results['multiplier'].unique())

    values = np.full((len(thresholds), len(multipliers)), np.nan)

    for _, row in sweep_results.iterrows():
        i = np.searchsorted(thresholds, row['threshold'])
        j = np.searchsorted(multipliers, row['multiplier'])
        if i < len(thresholds) and j < len(multipliers):
            values[i, j] = row[metric]

    return thresholds, multipliers, values


def rank_sweep_results(
    sweep_results: pd.DataFrame,
    primary_metric: str = "sortino_oos",
    secondary_metric: str = "avg_cost",
    ascending_secondary: bool = True,
    top_n: int = 10,
) -> pd.DataFrame:
    if sweep_results.empty:
        return sweep_results

    ranked = sweep_results.sort_values(
        [primary_metric, secondary_metric],
        ascending=[False, ascending_secondary]
    ).head(top_n)

    ranked = ranked.reset_index(drop=True)
    ranked.index.name = 'rank'
    ranked.index = ranked.index + 1                   

    return ranked
