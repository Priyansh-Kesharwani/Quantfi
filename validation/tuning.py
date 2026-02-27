"""
Phase 3 — Nested Cross-Validation & Hyperparameter Tuning.

Implements safe, non-leaking hyperparameter optimisation:

  Outer loop: walk-forward (chronological)
  Inner loop: purged K-fold inside each outer train set

Optimisers: grid search, random search, and Bayesian (scikit-optimize).

Objective  S(θ) = median_f(M_f) − λ_var · std_f(M_f)
  where M_f is the per-fold metric (Sortino by default).

All results are logged as a JSON trace for full reproducibility.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from itertools import product
import json
import logging
import time

from validation.metrics import (
    sortino_ratio, information_coefficient, forward_returns,
    compute_all_metrics,
)
from validation.kfold import _purged_kfold_splits

logger = logging.getLogger(__name__)

@dataclass
class TuningConfig:
    """Hyperparameter tuning configuration."""
    method: str = "random_search"
    n_trials: int = 50
    lambda_var: float = 0.5
    objective: str = "sortino"
    inner_n_splits: int = 5
    inner_embargo: int = 20
    search_space: Dict[str, List] = field(default_factory=dict)
    seed: int = 42

    @classmethod
    def from_dict(cls, d: Dict[str, Any], seed: int = 42) -> "TuningConfig":
        return cls(
            method=d.get("method", "random_search"),
            n_trials=d.get("n_trials", 50),
            lambda_var=d.get("lambda_var", 0.5),
            objective=d.get("objective", "sortino"),
            search_space=d.get("search_space", {}),
            seed=seed,
        )

@dataclass
class TuningTrialResult:
    """Result of a single hyperparameter trial."""
    trial_idx: int
    params: Dict[str, Any]
    fold_metrics: List[float]
    score: float
    median_metric: float
    std_metric: float
    elapsed_s: float

@dataclass
class TuningResult:
    """Aggregate tuning result."""
    symbol: str
    interval: str
    method: str
    n_trials: int
    best_params: Dict[str, Any]
    best_score: float
    trials: List[TuningTrialResult] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "method": self.method,
            "n_trials": self.n_trials,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "trials": [
                {
                    "trial_idx": t.trial_idx,
                    "params": t.params,
                    "fold_metrics": t.fold_metrics,
                    "score": t.score,
                    "median_metric": t.median_metric,
                    "std_metric": t.std_metric,
                    "elapsed_s": t.elapsed_s,
                }
                for t in self.trials
            ],
            "config": self.config,
        }

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

def _generate_grid(search_space: Dict[str, List]) -> List[Dict[str, Any]]:
    """Generate all combinations from search space."""
    keys = list(search_space.keys())
    values = list(search_space.values())
    combos = list(product(*values))
    return [dict(zip(keys, c)) for c in combos]

def _sample_random(
    search_space: Dict[str, List],
    n_trials: int,
    rng: np.random.RandomState,
) -> List[Dict[str, Any]]:
    """Sample random parameter combinations."""
    candidates = []
    keys = list(search_space.keys())
    for _ in range(n_trials):
        params = {}
        for k in keys:
            vals = search_space[k]
            params[k] = vals[rng.randint(len(vals))]
        candidates.append(params)
    return candidates

def _evaluate_inner_cv(
    df_train: pd.DataFrame,
    score_fn: Callable[[pd.DataFrame, Dict[str, Any]], Tuple[pd.Series, pd.Series]],
    params: Dict[str, Any],
    n_splits: int = 5,
    embargo: int = 20,
    objective: str = "sortino",
    entry_threshold: float = 70.0,
    exit_threshold: float = 70.0,
) -> Tuple[List[float], float, float]:
    """Evaluate a candidate θ on inner purged K-fold.

    Parameters
    ----------
    df_train : pd.DataFrame
        Outer training data.
    score_fn : callable
        Function(df, params) -> (entry_scores, exit_scores).
    params : dict
        Candidate hyperparameters.
    n_splits, embargo : int
        Inner CV configuration.
    objective : str
        Metric to optimise.

    Returns
    -------
    fold_metrics : list of float
    median_metric : float
    std_metric : float
    """
    n = len(df_train)
    splits = _purged_kfold_splits(n, n_splits, embargo)

    col_map = {c.lower(): c for c in df_train.columns}
    close_col = col_map.get("close", "close")
    prices = df_train[close_col]

    fold_metrics = []

    for train_idx, test_idx in splits:
        try:
            df_inner_train = df_train.iloc[train_idx]
            df_inner_test = df_train.iloc[test_idx]

            df_combined = pd.concat([df_inner_train, df_inner_test])
            entry_scores, exit_scores = score_fn(df_combined, params)

            test_index = df_inner_test.index
            entry_test = entry_scores.reindex(test_index)
            exit_test = exit_scores.reindex(test_index)
            prices_test = prices.reindex(test_index)

            period_ret = prices_test.pct_change().fillna(0)
            mask = entry_test > entry_threshold
            signal_ret = period_ret.copy()
            signal_ret[~mask] = 0.0

            if objective == "sortino":
                m = sortino_ratio(signal_ret)
            elif objective == "ic":
                fwd = forward_returns(prices_test, horizon=5)
                m = information_coefficient(entry_test, fwd)
            elif objective == "ir":
                mean_r = signal_ret.mean()
                std_r = signal_ret.std()
                m = (mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0.0
            else:
                m = sortino_ratio(signal_ret)

            if np.isnan(m) or np.isinf(m):
                m = -999.0

            fold_metrics.append(float(m))

        except Exception as e:
            logger.warning(f"Inner fold failed: {e}")
            fold_metrics.append(-999.0)

    if not fold_metrics:
        return [], -999.0, 0.0

    med = float(np.median(fold_metrics))
    std = float(np.std(fold_metrics))
    return fold_metrics, med, std

def run_tuning(
    df_train: pd.DataFrame,
    score_fn: Callable[[pd.DataFrame, Dict[str, Any]], Tuple[pd.Series, pd.Series]],
    config: TuningConfig,
    entry_threshold: float = 70.0,
    exit_threshold: float = 70.0,
    symbol: str = "UNKNOWN",
    interval: str = "1d",
) -> TuningResult:
    """Run hyperparameter tuning on a training set using inner CV.

    Parameters
    ----------
    df_train : pd.DataFrame
        Outer-loop training data.
    score_fn : callable
        Function(df, params_dict) -> (entry_scores, exit_scores).
    config : TuningConfig
        Tuning configuration.
    entry_threshold, exit_threshold : float
        Score thresholds for signal generation.
    symbol, interval : str
        Labels for logging.

    Returns
    -------
    TuningResult
    """
    rng = np.random.RandomState(config.seed)

    if config.method == "grid":
        candidates = _generate_grid(config.search_space)
    elif config.method == "random_search":
        candidates = _sample_random(config.search_space, config.n_trials, rng)
    elif config.method == "bayesian":
        try:
            candidates = _bayesian_search(
                df_train, score_fn, config, entry_threshold, exit_threshold,
            )
        except ImportError:
            logger.warning("scikit-optimize not available, falling back to random_search")
            candidates = _sample_random(config.search_space, config.n_trials, rng)
    else:
        candidates = _sample_random(config.search_space, config.n_trials, rng)

    logger.info(
        f"Tuning {symbol}/{interval}: {len(candidates)} candidates, "
        f"method={config.method}, inner_cv={config.inner_n_splits}-fold"
    )

    result = TuningResult(
        symbol=symbol,
        interval=interval,
        method=config.method,
        n_trials=len(candidates),
        best_params={},
        best_score=-np.inf,
        config={"search_space": config.search_space, "seed": config.seed},
    )

    for i, params in enumerate(candidates):
        t0 = time.time()

        fold_metrics, median_m, std_m = _evaluate_inner_cv(
            df_train, score_fn, params,
            n_splits=config.inner_n_splits,
            embargo=config.inner_embargo,
            objective=config.objective,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
        )

        score = median_m - config.lambda_var * std_m

        elapsed = time.time() - t0

        trial = TuningTrialResult(
            trial_idx=i,
            params=params,
            fold_metrics=fold_metrics,
            score=score,
            median_metric=median_m,
            std_metric=std_m,
            elapsed_s=elapsed,
        )
        result.trials.append(trial)

        if score > result.best_score:
            result.best_score = score
            result.best_params = params.copy()

        if (i + 1) % max(1, len(candidates) // 10) == 0:
            logger.info(
                f"  Trial {i + 1}/{len(candidates)}: "
                f"score={score:.4f}, best={result.best_score:.4f}"
            )

    logger.info(
        f"Tuning complete: best_score={result.best_score:.4f}, "
        f"best_params={result.best_params}"
    )
    return result

def _bayesian_search(
    df_train: pd.DataFrame,
    score_fn: Callable,
    config: TuningConfig,
    entry_threshold: float,
    exit_threshold: float,
) -> List[Dict[str, Any]]:
    """Bayesian optimisation via scikit-optimize (if available).

    Falls back to random search if skopt is not installed.
    Returns a list of candidate dicts (pre-evaluated by skopt).
    """
    from skopt import gp_minimize
    from skopt.space import Categorical

    keys = list(config.search_space.keys())
    dimensions = [Categorical(config.search_space[k]) for k in keys]

    def _objective(values):
        params = dict(zip(keys, values))
        fold_metrics, med, std = _evaluate_inner_cv(
            df_train, score_fn, params,
            n_splits=config.inner_n_splits,
            embargo=config.inner_embargo,
            objective=config.objective,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
        )
        score = med - config.lambda_var * std
        return -score

    result = gp_minimize(
        _objective,
        dimensions,
        n_calls=config.n_trials,
        random_state=config.seed,
    )

    candidates = [dict(zip(keys, x)) for x in result.x_iters]
    return candidates

def parameter_sensitivity(
    df: pd.DataFrame,
    score_fn: Callable[[pd.DataFrame, Dict[str, Any]], Tuple[pd.Series, pd.Series]],
    base_params: Dict[str, Any],
    param_name: str,
    param_values: List,
    n_splits: int = 5,
    embargo: int = 20,
    objective: str = "sortino",
    entry_threshold: float = 70.0,
) -> pd.DataFrame:
    """Compute sensitivity of objective to a single parameter.

    Parameters
    ----------
    df : pd.DataFrame
        Data to evaluate on.
    score_fn : callable
        Score function.
    base_params : dict
        Baseline parameters.
    param_name : str
        Parameter to vary.
    param_values : list
        Values to sweep.

    Returns
    -------
    pd.DataFrame
        Columns: param_value, median_metric, std_metric, score.
    """
    rows = []
    for val in param_values:
        params = base_params.copy()
        params[param_name] = val

        fold_metrics, med, std = _evaluate_inner_cv(
            df, score_fn, params,
            n_splits=n_splits,
            embargo=embargo,
            objective=objective,
            entry_threshold=entry_threshold,
        )

        score = med - 0.5 * std
        rows.append({
            "param_value": val,
            "median_metric": med,
            "std_metric": std,
            "score": score,
            "n_folds_valid": sum(1 for m in fold_metrics if m > -999),
        })

    return pd.DataFrame(rows)

def ablation_study(
    df: pd.DataFrame,
    score_fn: Callable[[pd.DataFrame, Dict[str, Any]], Tuple[pd.Series, pd.Series]],
    base_params: Dict[str, Any],
    components: List[str],
    n_splits: int = 5,
    embargo: int = 20,
    objective: str = "sortino",
    entry_threshold: float = 70.0,
) -> pd.DataFrame:
    """Leave-one-out component ablation study.

    For each component in `components`, set its contribution to zero
    (by adding a `disable_{component}=True` flag to params) and
    measure performance degradation.

    Parameters
    ----------
    components : list of str
        Component names to ablate (e.g. ["ofi", "hawkes", "ldc"]).

    Returns
    -------
    pd.DataFrame
        Columns: component, baseline_score, ablated_score, impact.
    """
    _, base_med, base_std = _evaluate_inner_cv(
        df, score_fn, base_params,
        n_splits=n_splits, embargo=embargo,
        objective=objective, entry_threshold=entry_threshold,
    )
    base_score = base_med - 0.5 * base_std

    rows = [{"component": "baseline", "score": base_score, "impact": 0.0}]

    for comp in components:
        ablated_params = base_params.copy()
        ablated_params[f"disable_{comp}"] = True

        _, abl_med, abl_std = _evaluate_inner_cv(
            df, score_fn, ablated_params,
            n_splits=n_splits, embargo=embargo,
            objective=objective, entry_threshold=entry_threshold,
        )
        abl_score = abl_med - 0.5 * abl_std
        impact = base_score - abl_score

        rows.append({
            "component": comp,
            "score": abl_score,
            "impact": impact,
        })

    return pd.DataFrame(rows)
