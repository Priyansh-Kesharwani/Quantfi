"""
Multi-Fidelity Bayesian Optimization (MFBO).

Search space + fidelity levels; surrogate GP (Optuna TPE/GP); EI per cost;
trial persistence to tuning_trials.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    optuna = None
    TPESampler = None


def _suggest_params(trial: Any, search_space: Dict[str, Any]) -> Dict[str, Any]:
    config = {}
    for name, spec in search_space.items():
        if isinstance(spec, (list, tuple)):
            if len(spec) == 2 and isinstance(spec[0], (int, float)) and isinstance(spec[1], (int, float)):
                if isinstance(spec[0], int) and isinstance(spec[1], int):
                    config[name] = trial.suggest_int(name, int(spec[0]), int(spec[1]))
                else:
                    config[name] = trial.suggest_float(name, float(spec[0]), float(spec[1]))
            else:
                config[name] = trial.suggest_categorical(name, list(spec))
        elif isinstance(spec, dict):
            low = spec.get("low", spec.get("min", 0))
            high = spec.get("high", spec.get("max", 1))
            if spec.get("log", False):
                config[name] = trial.suggest_float(name, low, high, log=True)
            else:
                config[name] = trial.suggest_float(name, low, high)
        else:
            config[name] = trial.suggest_categorical(name, [spec])
    return config


def run_mfbo(
    objective_fn: Callable[[Dict[str, Any], str], float],
    search_space: Dict[str, Any],
    fidelity_spec: Dict[str, Dict[str, Any]],
    n_trials: int,
    seed: int,
    output_path: Path,
    *,
    cost_key: str = "cost",
    maximize: bool = True,
) -> Dict[str, Any]:
    """
    Run multi-fidelity Bayesian optimization.

    Parameters
    ----------
    objective_fn : callable
        (config, fidelity) -> score. Fidelity is one of fidelity_spec keys.
    search_space : dict
        Param name -> list of values (categorical) or [low, high] (numeric).
    fidelity_spec : dict
        Fidelity name -> {"cost": float, ...}. Cost used for EI/cost (score/cost).
    n_trials : int
        Total number of trials.
    seed : int
        Random seed for reproducibility.
    output_path : Path
        Directory or file; trials appended to tuning_trials.json under output_path.
    cost_key : str
        Key in each fidelity spec for evaluation cost (default "cost").
    maximize : bool
        If True, maximize objective; else minimize.

    Returns
    -------
    dict
        best_params, best_value, best_fidelity, trials (list of recorded trials).
    """
    if optuna is None or TPESampler is None:
        raise RuntimeError("optuna is required for run_mfbo; install with: pip install optuna")

    fidelity_names = list(fidelity_spec.keys())
    if not fidelity_names:
        fidelity_names = ["low"]
        fidelity_spec = {"low": {cost_key: 1.0}}

    path = Path(output_path)
    if path.suffix:
        trials_file = path
    else:
        path.mkdir(parents=True, exist_ok=True)
        trials_file = path / "tuning_trials.json"

    all_trials: List[Dict[str, Any]] = []
    best_value: Optional[float] = None
    best_params: Optional[Dict[str, Any]] = None
    best_fidelity: Optional[str] = None

    def objective(trial: Any) -> float:
        config = _suggest_params(trial, search_space)
        fidelity = trial.suggest_categorical("_fidelity", fidelity_names)
        cost = float(fidelity_spec.get(fidelity, {}).get(cost_key, 1.0))
        if cost <= 0:
            cost = 1.0
        t0 = time.perf_counter()
        score = objective_fn(config, fidelity)
        elapsed = time.perf_counter() - t0
        rec = {
            "trial_id": trial.number,
            "fidelity": fidelity,
            "score": score,
            "config": config,
            "compute_time_s": round(elapsed, 4),
        }
        all_trials.append(rec)
        with open(trials_file, "w") as f:
            json.dump(all_trials, f, indent=2, default=str)
        value = score / cost if maximize else -score / cost
        return value

    sampler = TPESampler(seed=seed, n_startup_trials=min(10, n_trials // 2))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_trial = study.best_trial
    if best_trial is not None and all_trials:
        for t in all_trials:
            if t["trial_id"] == best_trial.number:
                best_value = t["score"]
                best_params = t["config"]
                best_fidelity = t["fidelity"]
                break

    return {
        "best_params": best_params or {},
        "best_value": best_value,
        "best_fidelity": best_fidelity,
        "n_trials": len(all_trials),
        "trials": all_trials,
    }
