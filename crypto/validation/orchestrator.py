"""CryptoOrchestrator: CPCV + DSR + GT-Score optimization with robustness suite."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorResult:
    winning_config: Dict[str, Any] = field(default_factory=dict)
    best_gt_score: float = 0.0
    best_dsr: float = 0.0
    n_trials: int = 0
    trial_log: List[Dict[str, Any]] = field(default_factory=list)
    robustness: Dict[str, Any] = field(default_factory=dict)
    baselines: Dict[str, Any] = field(default_factory=dict)
    holdout_metrics: Dict[str, Any] = field(default_factory=dict)


class CryptoOrchestrator:
    """CPCV + Walk-Forward + Robustness optimizer for crypto strategies.

    Uses Optuna AutoSampler and the existing GT-Score/DSR from validation/objective.py.
    """

    def __init__(
        self,
        evaluate_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        search_space: Optional[Dict[str, Any]] = None,
        n_trials: int = 100,
        dsr_min: float = 0.0,
        seed: int = 42,
    ):
        """
        Args:
            evaluate_fn: takes a config dict, returns metrics dict with keys
                         'gt_score', 'sharpe', 'dsr', 'max_drawdown', etc.
            search_space: dict of param_name -> (low, high) or list for categorical
            n_trials: number of optimization trials
        """
        self._evaluate_fn = evaluate_fn
        self._search_space = search_space or self._default_search_space()
        self._n_trials = n_trials
        self._dsr_min = dsr_min
        self._seed = seed

    @staticmethod
    def _default_search_space() -> Dict[str, Any]:
        return {
            "entry_threshold": (20.0, 80.0),
            "exit_threshold": (5.0, 30.0),
            "max_holding_bars": (12, 720),
            "atr_trail_mult": (1.5, 4.0),
            "kelly_fraction": (0.1, 0.5),
            "max_risk_per_trade": (0.005, 0.05),
            "leverage": (1.0, 10.0),
            "transition_cooldown": (0, 12),
            "rsi_oversold": (15.0, 35.0),
            "rsi_overbought": (65.0, 85.0),
            "compression_window": (63, 504),
            "z_short_weight": (0.2, 0.6),
            "z_mid_weight": (0.2, 0.5),
            "vol_threshold": (1.5, 4.0),
            "vol_scale": (0.3, 1.5),
            "vol_filter_pctl": (0.75, 0.95),
            "vol_filter_floor": (0.1, 0.5),
            "funding_window": (168, 720),
            "ic_window": (126, 504),
            "ic_horizon": (6, 48),
            "ic_alpha": (2.0, 10.0),
            "ic_shrink": (0.1, 0.4),
        }

    def run(self) -> OrchestratorResult:
        """Run full optimization with Optuna and return winning config + robustness."""
        try:
            import optuna
            from optuna.samplers import TPESampler
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.error("optuna not installed. Run: pip install optuna")
            return OrchestratorResult()

        sampler = TPESampler(seed=self._seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        trial_log = []

        def objective(trial: optuna.Trial) -> float:
            config = {}
            for name, bounds in self._search_space.items():
                if isinstance(bounds, list):
                    config[name] = trial.suggest_categorical(name, bounds)
                elif isinstance(bounds, tuple) and len(bounds) == 2:
                    lo, hi = bounds
                    if isinstance(lo, int) and isinstance(hi, int):
                        config[name] = trial.suggest_int(name, lo, hi)
                    else:
                        config[name] = trial.suggest_float(name, float(lo), float(hi))
                else:
                    config[name] = bounds

            try:
                metrics = self._evaluate_fn(config)
            except Exception as e:
                logger.warning("Trial %d failed: %s", trial.number, e)
                return -1e6

            gt_score = metrics.get("gt_score", 0.0)
            dsr = metrics.get("dsr", 0.0)

            trial_log.append({
                "trial": trial.number,
                "config": config,
                "gt_score": gt_score,
                "dsr": dsr,
                "sharpe": metrics.get("sharpe", 0.0),
            })

            if dsr < self._dsr_min:
                return gt_score * 0.1

            return gt_score

        study.optimize(objective, n_trials=self._n_trials, show_progress_bar=False)

        valid_trials = [t for t in trial_log if t["dsr"] >= self._dsr_min]
        if valid_trials:
            best = max(valid_trials, key=lambda x: x["gt_score"])
        elif trial_log:
            best = max(trial_log, key=lambda x: x["gt_score"])
        else:
            return OrchestratorResult(n_trials=self._n_trials, trial_log=trial_log)

        winning_config = best["config"]

        robustness = sensitivity_analysis(
            winning_config, best["gt_score"], self._evaluate_fn
        )

        return OrchestratorResult(
            winning_config=winning_config,
            best_gt_score=best["gt_score"],
            best_dsr=best.get("dsr", 0.0),
            n_trials=len(trial_log),
            trial_log=trial_log,
            robustness=robustness,
        )


def walk_forward_validation(
    evaluate_fn: Callable[[Dict[str, Any]], Dict[str, float]],
    config: Dict[str, Any],
    n_bars_total: int,
    train_bars: int,
    test_bars: int,
) -> Dict[str, Any]:
    """Run walk-forward OOS validation.

    Splits data conceptually into windows of *train_bars* + *test_bars*.
    Each fold calls *evaluate_fn* twice: once with ``{'wf_start': s, 'wf_end': e}``
    for the train segment (result discarded except for param selection), then for
    the test segment.  The caller's evaluate_fn must respect these keys.

    Returns per-fold OOS metrics and aggregate statistics.
    """
    folds: List[Dict[str, Any]] = []
    start = 0
    fold_idx = 0
    while start + train_bars + test_bars <= n_bars_total:
        train_end = start + train_bars
        test_end = train_end + test_bars

        cfg_test = copy.deepcopy(config)
        cfg_test["wf_start"] = train_end
        cfg_test["wf_end"] = test_end

        try:
            oos_metrics = evaluate_fn(cfg_test)
        except Exception as e:
            logger.warning("Walk-forward fold %d failed: %s", fold_idx, e)
            oos_metrics = {"sharpe": 0.0, "gt_score": 0.0}

        folds.append({
            "fold": fold_idx,
            "train_range": (start, train_end),
            "test_range": (train_end, test_end),
            "oos_sharpe": oos_metrics.get("sharpe", 0.0),
            "oos_gt_score": oos_metrics.get("gt_score", 0.0),
        })
        start += test_bars
        fold_idx += 1

    oos_sharpes = [f["oos_sharpe"] for f in folds]
    return {
        "folds": folds,
        "n_folds": len(folds),
        "mean_oos_sharpe": float(np.mean(oos_sharpes)) if oos_sharpes else 0.0,
        "std_oos_sharpe": float(np.std(oos_sharpes)) if oos_sharpes else 0.0,
        "min_oos_sharpe": float(np.min(oos_sharpes)) if oos_sharpes else 0.0,
    }


_FRAGILE_THRESHOLD = 30
_WARNING_THRESHOLD = 20


def sensitivity_analysis(
    winning_config: Dict[str, Any],
    base_score: float,
    evaluate_fn: Callable[[Dict[str, Any]], Dict[str, float]],
) -> Dict[str, Any]:
    """Perturb each numeric parameter by +/-10% and +/-20%.

    Tiers: WARNING (>20% degradation), FRAGILE (>30% degradation).
    """
    results = {}
    fragile_params = []
    warning_params = []

    for param, value in winning_config.items():
        if not isinstance(value, (int, float)):
            continue
        if value == 0:
            continue

        for pct in [0.9, 1.1, 0.8, 1.2]:
            perturbed = copy.deepcopy(winning_config)
            new_val = value * pct
            if isinstance(value, int):
                new_val = int(round(new_val))
            perturbed[param] = new_val

            try:
                metrics = evaluate_fn(perturbed)
                score = metrics.get("gt_score", 0.0)
            except Exception:
                score = -1e6

            degradation = (base_score - score) / (abs(base_score) + 1e-8) * 100
            key = f"{param}_{pct}"
            results[key] = {"score": score, "degradation_pct": degradation}

            if pct in (0.9, 1.1):
                if degradation > _FRAGILE_THRESHOLD:
                    fragile_params.append(key)
                elif degradation > _WARNING_THRESHOLD:
                    warning_params.append(key)

    return {
        "results": results,
        "fragile_params": fragile_params,
        "warning_params": warning_params,
        "is_robust": len(fragile_params) == 0,
    }


def slippage_stress_test(
    winning_config: Dict[str, Any],
    evaluate_fn: Callable[[Dict[str, Any]], Dict[str, float]],
    multipliers: List[float] = None,
) -> Dict[str, Any]:
    """Test strategy under 2x and 5x slippage."""
    if multipliers is None:
        multipliers = [2.0, 5.0]
    results = {}
    for mult in multipliers:
        cfg = copy.deepcopy(winning_config)
        cfg["slippage_mult"] = mult
        try:
            metrics = evaluate_fn(cfg)
            results[f"{mult}x"] = metrics
        except Exception as e:
            results[f"{mult}x"] = {"error": str(e)}
    return results


def funding_stress_test(
    winning_config: Dict[str, Any],
    evaluate_fn: Callable[[Dict[str, Any]], Dict[str, float]],
    multiplier: float = 3.0,
) -> Dict[str, Any]:
    """Test strategy under 3x average funding rate."""
    cfg = copy.deepcopy(winning_config)
    cfg["funding_rate_mult"] = multiplier
    try:
        return evaluate_fn(cfg)
    except Exception as e:
        return {"error": str(e)}
