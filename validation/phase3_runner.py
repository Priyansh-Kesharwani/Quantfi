"""
Phase 3 — Orchestrator for Real-World Validation & Parameter Tuning.

Coordinates:
  1. Data loading & integrity checks
  2. Walk-forward outer loop with nested inner-CV tuning
  3. Out-of-sample evaluation with execution costs
  4. Hawkes stress testing
  5. Robustness analysis (subsampling, ablation, threshold sweep)
  6. HTML report generation

Usage:
    python -m validation.phase3_runner \\
        --assets AAPL,SPY,GLD \\
        --intervals 1d \\
        --tuning random_search \\
        --trials 50 \\
        --out validation/outputs/
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
import time
import yaml
import hashlib
import sys
import argparse

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from validation.data_integrity import (
    validate_dataframe, canonicalize, fetch_and_validate,
)
from validation.walkforward import walkforward_cv, WalkForwardConfig, WalkForwardResult
from validation.kfold import purged_kfold, PurgedKFoldConfig, KFoldResult
from validation.metrics import (
    compute_all_metrics, sortino_ratio, information_coefficient,
    forward_returns, max_drawdown, cagr,
)
from validation.execution_model import (
    ExecutionConfig, apply_execution_costs, slippage_sensitivity_matrix,
)
from validation.tuning import (
    TuningConfig, TuningResult, run_tuning,
    parameter_sensitivity, ablation_study,
)
from simulations.hawkes_simulator import (
    run_all_regimes, validate_estimation,
    generate_synthetic_lob, generate_synthetic_trades,
)

@dataclass
class Phase3Config:
    """Full Phase 3 configuration."""
    seed: int = 42
    assets: List[Dict[str, Any]] = field(default_factory=list)
    walkforward: Dict[str, Any] = field(default_factory=dict)
    inner_cv: Dict[str, Any] = field(default_factory=dict)
    tuning: Dict[str, Any] = field(default_factory=dict)
    execution: Dict[str, Any] = field(default_factory=dict)
    hawkes_stress: Dict[str, Any] = field(default_factory=dict)
    robustness: Dict[str, Any] = field(default_factory=dict)
    scoring: Dict[str, Any] = field(default_factory=dict)
    reporting: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str = "config/phase3.yml") -> "Phase3Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(
            seed=raw.get("determinism", {}).get("rng_seed", 42),
            assets=raw.get("data", {}).get("assets", []),
            walkforward=raw.get("walkforward", {}),
            inner_cv=raw.get("inner_cv", {}),
            tuning=raw.get("tuning", {}),
            execution=raw.get("execution", {}),
            hawkes_stress=raw.get("hawkes_stress", {}),
            robustness=raw.get("robustness", {}),
            scoring=raw.get("scoring", {}),
            reporting=raw.get("reporting", {}),
        )

def run_asset_validation(
    df: pd.DataFrame,
    score_fn: Callable[[pd.DataFrame, Dict[str, Any]], Tuple[pd.Series, pd.Series]],
    symbol: str,
    interval: str,
    config: Phase3Config,
    output_dir: str = "validation/outputs",
) -> Dict[str, Any]:
    """Run full Phase 3 validation for a single asset/interval.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical OHLCV data.
    score_fn : callable
        Function(df, params) -> (entry_scores, exit_scores).
    symbol : str
        Asset symbol.
    interval : str
        Data interval.
    config : Phase3Config
    output_dir : str

    Returns
    -------
    dict
        Full results including tuning, OOS metrics, stress tests.
    """
    np.random.seed(config.seed)
    t0 = time.time()

    results: Dict[str, Any] = {
        "symbol": symbol,
        "interval": interval,
        "seed": config.seed,
        "n_bars": len(df),
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    wf_cfg = WalkForwardConfig.from_dict(config.walkforward)
    entry_thr = config.scoring.get("entry_threshold", 70.0)
    exit_thr = config.scoring.get("exit_threshold", 70.0)
    fwd_horizons = config.scoring.get("forward_horizons", [5, 10, 20])

    logger.info(f"[{symbol}/{interval}] Step 1: Walk-forward + tuning")

    tuning_cfg = TuningConfig.from_dict(config.tuning, seed=config.seed)
    tuning_cfg.inner_n_splits = config.inner_cv.get("n_splits", 5)
    tuning_cfg.inner_embargo = config.inner_cv.get("embargo_bars", 20)

    from validation.walkforward import _generate_folds
    folds = _generate_folds(
        len(df), wf_cfg.train_window, wf_cfg.test_window,
        overlap=wf_cfg.overlap, expanding=wf_cfg.expanding,
    )

    oos_fold_results = []
    tuning_results = []

    for fold_i, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        logger.info(f"  Outer fold {fold_i}: train=[{tr_s}:{tr_e}], test=[{te_s}:{te_e}]")

        df_train = df.iloc[tr_s:tr_e]
        df_test = df.iloc[te_s:te_e]

        if config.tuning.get("search_space"):
            tune_result = run_tuning(
                df_train, score_fn, tuning_cfg,
                entry_threshold=entry_thr, exit_threshold=exit_thr,
                symbol=symbol, interval=interval,
            )
            best_params = tune_result.best_params
            tuning_results.append(tune_result.to_dict())
        else:
            best_params = {}
            tuning_results.append({"fold": fold_i, "params": {}, "note": "no search space"})

        try:
            df_combined = pd.concat([df_train, df_test])
            entry_scores, exit_scores = score_fn(df_combined, best_params)

            test_idx = df_test.index
            entry_test = entry_scores.reindex(test_idx)
            exit_test = exit_scores.reindex(test_idx)

            col_map = {c.lower(): c for c in df_test.columns}
            prices_test = df_test[col_map.get("close", "close")]

            oos_metrics = compute_all_metrics(
                entry_test, exit_test, prices_test,
                forward_horizons=fwd_horizons,
                entry_threshold=entry_thr, exit_threshold=exit_thr,
            )

            period_ret = prices_test.pct_change().fillna(0)
            vol_col = col_map.get("volume", "volume")
            volumes = df_test[vol_col] if vol_col in df_test.columns else pd.Series(
                np.ones(len(df_test)), index=df_test.index
            )

            exec_cfg = ExecutionConfig.from_dict(config.execution)
            adj_ret, cost_report = apply_execution_costs(
                period_ret, entry_test, exit_test, volumes,
                config=exec_cfg, entry_threshold=entry_thr,
                exit_threshold=exit_thr, seed=config.seed + fold_i,
            )

            oos_fold_results.append({
                "fold_idx": fold_i,
                "params": best_params,
                "metrics": oos_metrics,
                "execution_costs": cost_report,
                "test_start": str(df_test.index[0]),
                "test_end": str(df_test.index[-1]),
                "test_size": len(df_test),
            })

        except Exception as e:
            logger.error(f"  Fold {fold_i} OOS evaluation failed: {e}")
            oos_fold_results.append({
                "fold_idx": fold_i,
                "error": str(e),
            })

    results["oos_folds"] = oos_fold_results
    results["tuning_traces"] = tuning_results

    logger.info(f"[{symbol}/{interval}] Step 2: Aggregating OOS metrics")
    results["oos_summary"] = _aggregate_oos(oos_fold_results, fwd_horizons)

    logger.info(f"[{symbol}/{interval}] Step 3: Hawkes stress tests")
    if config.hawkes_stress.get("regimes"):
        hawkes_results = run_all_regimes(
            config.hawkes_stress["regimes"],
            base_seed=config.seed,
        )
        hawkes_validations = [
            validate_estimation(
                r["true_intensity"], r,
                rmse_threshold=config.hawkes_stress.get("rmse_threshold", 0.10),
            )
            for r in hawkes_results
        ]
        results["hawkes_stress"] = {
            "regimes": [
                {k: v for k, v in r.items()
                 if k not in ("events", "grid", "true_intensity")}
                for r in hawkes_results
            ],
            "validations": hawkes_validations,
        }

    logger.info(f"[{symbol}/{interval}] Step 4: Slippage sensitivity")
    try:
        col_map = {c.lower(): c for c in df.columns}
        prices = df[col_map.get("close", "close")]
        period_ret = prices.pct_change().fillna(0)
        volumes = df[col_map.get("volume", "volume")]

        entry_full, exit_full = score_fn(df, best_params if tuning_results else {})
        slip_matrix = slippage_sensitivity_matrix(
            period_ret, entry_full, exit_full, volumes,
            entry_threshold=entry_thr, exit_threshold=exit_thr,
            seed=config.seed,
        )
        results["slippage_matrix"] = slip_matrix.to_dict()
    except Exception as e:
        logger.warning(f"Slippage sensitivity failed: {e}")
        results["slippage_matrix"] = {"error": str(e)}

    logger.info(f"[{symbol}/{interval}] Step 5: Robustness tests")
    robustness = config.robustness

    if robustness.get("threshold_sweep"):
        threshold_results = _threshold_sweep(
            df, score_fn, best_params if tuning_results else {},
            robustness["threshold_sweep"], fwd_horizons,
        )
        results["threshold_sweep"] = threshold_results

    if robustness.get("subsample_n_trials"):
        subsample_results = _subsample_stability(
            df, score_fn, best_params if tuning_results else {},
            n_trials=robustness.get("subsample_n_trials", 20),
            fraction=robustness.get("subsample_fraction", 0.5),
            entry_threshold=entry_thr,
            seed=config.seed,
        )
        results["subsample_stability"] = subsample_results

    results["elapsed_s"] = time.time() - t0
    results_path = out / f"{symbol}_{interval}_tuning.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    results["checksum"] = _compute_checksum(results_path)

    return results

def _aggregate_oos(
    fold_results: List[Dict[str, Any]],
    forward_horizons: List[int],
) -> Dict[str, Any]:
    """Aggregate out-of-sample fold metrics."""
    summary: Dict[str, Any] = {}

    valid_folds = [f for f in fold_results if "metrics" in f]
    if not valid_folds:
        return {"note": "no valid OOS folds"}

    sortinos = []
    cagrs = []
    dds = []
    ics = {}

    for f in valid_folds:
        em = f["metrics"].get("entry_metrics", {})
        sortinos.append(em.get("sortino", np.nan))
        cagrs.append(em.get("cagr", np.nan))
        dds.append(em.get("max_drawdown", np.nan))

        for h in forward_horizons:
            key = f"ic_{h}d"
            ics.setdefault(key, []).append(em.get(key, np.nan))

    summary["median_sortino"] = float(np.nanmedian(sortinos))
    summary["std_sortino"] = float(np.nanstd(sortinos))
    summary["median_cagr"] = float(np.nanmedian(cagrs))
    summary["median_drawdown"] = float(np.nanmedian(dds))

    for key, vals in ics.items():
        summary[f"median_{key}"] = float(np.nanmedian(vals))
        summary[f"std_{key}"] = float(np.nanstd(vals))

    costs = [f.get("execution_costs", {}) for f in valid_folds]
    avg_erosion = np.nanmean([c.get("pnl_erosion", 0) for c in costs])
    summary["avg_pnl_erosion"] = float(avg_erosion)
    summary["n_valid_folds"] = len(valid_folds)

    return summary

def _threshold_sweep(
    df: pd.DataFrame,
    score_fn: Callable,
    params: Dict[str, Any],
    thresholds: List[float],
    forward_horizons: List[int],
) -> List[Dict[str, Any]]:
    """Sweep entry thresholds and report trade count vs ROI."""
    col_map = {c.lower(): c for c in df.columns}
    prices = df[col_map.get("close", "close")]

    entry_scores, exit_scores = score_fn(df, params)
    period_ret = prices.pct_change().fillna(0)

    results = []
    for thr in thresholds:
        mask = entry_scores > thr
        sig_ret = period_ret.copy()
        sig_ret[~mask] = 0.0

        equity = (1 + sig_ret).cumprod()
        s = sortino_ratio(sig_ret)
        dd = max_drawdown(equity)
        c = cagr(equity)
        n_signals = int(mask.sum())

        fwd = forward_returns(prices, horizon=5)
        ic = information_coefficient(entry_scores, fwd)

        results.append({
            "threshold": thr,
            "n_signals": n_signals,
            "pct_active": float(mask.mean()),
            "sortino": s,
            "cagr": c,
            "max_drawdown": dd,
            "ic_5d": ic,
        })

    return results

def _subsample_stability(
    df: pd.DataFrame,
    score_fn: Callable,
    params: Dict[str, Any],
    n_trials: int = 20,
    fraction: float = 0.5,
    entry_threshold: float = 70.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Random subsampling stability test."""
    rng = np.random.RandomState(seed)
    col_map = {c.lower(): c for c in df.columns}
    prices = df[col_map.get("close", "close")]

    sortinos = []
    ics = []

    n = len(df)
    sample_size = int(n * fraction)

    for trial in range(n_trials):
        start = rng.randint(0, n - sample_size)
        df_sub = df.iloc[start:start + sample_size]
        prices_sub = prices.iloc[start:start + sample_size]

        try:
            entry_sub, exit_sub = score_fn(df_sub, params)
            period_ret = prices_sub.pct_change().fillna(0)
            mask = entry_sub > entry_threshold
            sig_ret = period_ret.copy()
            sig_ret[~mask] = 0.0
            s = sortino_ratio(sig_ret)
            sortinos.append(s)

            fwd = forward_returns(prices_sub, horizon=5)
            ic = information_coefficient(entry_sub, fwd)
            ics.append(ic)
        except Exception:
            pass

    return {
        "n_trials": n_trials,
        "fraction": fraction,
        "sortino_mean": float(np.nanmean(sortinos)) if sortinos else np.nan,
        "sortino_std": float(np.nanstd(sortinos)) if sortinos else np.nan,
        "ic_mean": float(np.nanmean(ics)) if ics else np.nan,
        "ic_std": float(np.nanstd(ics)) if ics else np.nan,
    }

def _compute_checksum(path: str) -> str:
    """Compute SHA256 of a file for determinism verification."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    """Command-line entry point for Phase 3 runner."""
    parser = argparse.ArgumentParser(description="Phase 3 — Validation & Tuning Runner")
    parser.add_argument("--config", default="config/phase3.yml", help="Config YAML path")
    parser.add_argument("--assets", default=None, help="Comma-separated asset symbols (overrides config)")
    parser.add_argument("--intervals", default=None, help="Comma-separated intervals (overrides config)")
    parser.add_argument("--tuning", default=None, help="Tuning method (overrides config)")
    parser.add_argument("--trials", type=int, default=None, help="Number of tuning trials")
    parser.add_argument("--out", default="validation/outputs", help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    config = Phase3Config.from_yaml(args.config)

    if args.tuning:
        config.tuning["method"] = args.tuning
    if args.trials:
        config.tuning["n_trials"] = args.trials

    if args.assets:
        asset_syms = [s.strip() for s in args.assets.split(",")]
    else:
        asset_syms = [a["symbol"] for a in config.assets]

    logger.info(f"Phase 3 runner: assets={asset_syms}")
    logger.info(f"Config seed: {config.seed}")

    def _default_score_fn(df, params=None):
        np.random.seed(config.seed)
        n = len(df)
        entry = pd.Series(
            np.clip(50 + np.random.randn(n) * 15, 0, 100),
            index=df.index, name="Entry_Score",
        )
        exit_ = pd.Series(
            np.clip(50 + np.random.randn(n) * 15, 0, 100),
            index=df.index, name="Exit_Score",
        )
        return entry, exit_

    for sym in asset_syms:
        asset_cfg = next((a for a in config.assets if a["symbol"] == sym), None)
        intervals = asset_cfg.get("intervals", ["1d"]) if asset_cfg else ["1d"]

        if args.intervals:
            intervals = [i.strip() for i in args.intervals.split(",")]

        for interval in intervals:
            logger.info(f"=== Running {sym}/{interval} ===")

            df, fetch_report = fetch_and_validate(
                sym, save_csv=False,
                min_history_years=asset_cfg.get("min_history_years", 10) if asset_cfg else 10,
            )

            if df is None:
                logger.error(f"Failed to load {sym}: {fetch_report}")
                continue

            result = run_asset_validation(
                df, _default_score_fn, sym, interval,
                config, output_dir=args.out,
            )

            logger.info(
                f"  {sym}/{interval} complete: "
                f"{result.get('oos_summary', {}).get('n_valid_folds', 0)} OOS folds, "
                f"elapsed={result.get('elapsed_s', 0):.1f}s"
            )

    logger.info("Phase 3 runner complete.")

if __name__ == "__main__":
    main()
