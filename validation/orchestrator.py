"""
Orchestrator: data → CPCV → backtest → GT-Score/DSR → winning config.

Pipeline: ingest data, generate CPCV splits, run backtest per split per config,
evaluate GT-Score and DSR, persist winning_config.json and artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from validation.objective import (
    compute_dsr,
    compute_gt_score,
    equity_curve_from_result,
)
from validation.validator import CPCVConfig, CPCVSplit, generate_cpcv_splits

def _ts_to_iso(ts) -> str:
    """Convert timestamp to ISO date string for metadata."""
    if hasattr(ts, "to_pydatetime"):
        return ts.to_pydatetime().date().isoformat()
    if hasattr(ts, "isoformat"):
        return ts.isoformat()[:10]
    return str(ts)[:10]

@dataclass
class OrchestratorResult:
    winning_config: Optional[Dict[str, Any]] = None
    winning_dsr: Optional[float] = None
    winning_mean_gt: Optional[float] = None
    gt_scores: Dict[str, Any] = field(default_factory=dict)
    dsr_by_config: Dict[str, float] = field(default_factory=dict)
    n_trials: int = 0
    artifact_paths: Dict[str, str] = field(default_factory=dict)

def _load_data(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    asset_meta: Optional[Dict[str, Dict[str, str]]] = None,
) -> Tuple[pd.DatetimeIndex, Any]:
    from backtester.portfolio_simulator import prepare_multi_asset_data

    meta = asset_meta or {sym: {"asset_type": "equity", "currency": "USD"} for sym in symbols}
    date_index, assets = prepare_multi_asset_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        asset_meta=meta,
    )
    return date_index, assets

def _config_to_sim_config(
    config: Dict[str, Any],
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    *,
    cost_free: bool = False,
) -> Any:
    from backtester.portfolio_simulator import ExitParams, SimConfig

    exit_params = ExitParams(
        atr_init_mult=float(config.get("atr_init_mult", 2.0)),
        atr_trail_mult=float(config.get("atr_trail_mult", 2.5)),
        min_stop_pct=float(config.get("min_stop_pct", 4.0)),
        score_rel_mult=float(config.get("score_rel_mult", 0.4)),
        score_abs_floor=float(config.get("score_abs_floor", 35.0)),
        max_holding_days=int(config.get("max_holding_days", 30)),
        use_atr_stop=bool(config.get("use_atr_stop", True)),
        min_holding_days=int(config.get("min_holding_days", 0)),
        vol_regime_stop_widen=float(config.get("vol_regime_stop_widen", 1.5)),
    )
    return SimConfig(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=float(config.get("initial_capital", 100_000.0)),
        entry_score_threshold=float(config.get("entry_score_threshold", 70.0)),
        entry_confirmation_bars=int(config.get("entry_confirmation_bars", 1)),
        exit_params=exit_params,
        max_positions=int(config.get("max_positions", 10)),
        use_score_weighting=bool(config.get("use_score_weighting", True)),
        slippage_bps=float(config.get("slippage_bps", 5.0)),
        run_benchmarks=True,
        cost_free=cost_free,
        cost_class_override=config.get("cost_class", None),
        min_invested_fraction=float(config.get("min_invested_fraction", 0.0)),
        scoring_mode=config.get("scoring_mode", "mean_reversion"),
    )

def _config_to_alloc_config(
    config: Dict[str, Any],
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    *,
    cost_free: bool = False,
) -> Any:
    from backtester.portfolio_simulator import AllocationConfig

    return AllocationConfig(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=float(config.get("initial_capital", 100_000.0)),
        risk_on_equity_pct=float(config.get("risk_on_equity_pct", 0.85)),
        risk_off_equity_pct=float(config.get("risk_off_equity_pct", 0.40)),
        theta_tilt=float(config.get("theta_tilt", 2.0)),
        min_weight_floor=float(config.get("min_weight_floor", 0.05)),
        rebalance_freq_days=int(config.get("rebalance_freq_days", 21)),
        min_rebalance_delta=float(config.get("min_rebalance_delta", 0.05)),
        jump_penalty=float(config.get("jump_penalty", 25.0)),
        regime_n_states=int(config.get("regime_n_states", 2)),
        regime_window=int(config.get("regime_window", 504)),
        regime_refit_every=int(config.get("regime_refit_every", 63)),
        drawdown_circuit_threshold=float(config.get("drawdown_circuit_threshold", -0.10)),
        hysteresis_enter=float(config.get("hysteresis_enter", 0.70)),
        hysteresis_exit=float(config.get("hysteresis_exit", 0.60)),
        cooldown_days=int(config.get("cooldown_days", 15)),
        slippage_bps=0.0 if cost_free else float(config.get("slippage_bps", 5.0)),
        cost_free=cost_free,
        cash_return_annual=float(config.get("cash_return_annual", 0.02)),
        scoring_mode=config.get("scoring_mode", "adaptive"),
        run_benchmarks=True,
    )

def _slice_assets_for_split(
    assets: Any,
    test_start: int,
    test_end: int,
) -> Dict[str, Any]:
    """Slice each asset's arrays to [test_start, test_end] for bot backtest."""
    from backtester.portfolio_simulator import AssetData

    sliced = {}
    slice_len = test_end - test_start + 1
    for sym, ad in assets.items():
        end_inclusive = test_end + 1
        fvi_orig = getattr(ad, "first_valid_idx", 0)
        fvi_sliced = max(0, fvi_orig - test_start)
        first_valid_idx = min(fvi_sliced, slice_len - 2) if slice_len >= 2 else 0
        trend_sc = ad.trend_score[test_start:end_inclusive] if getattr(ad, "trend_score", None) is not None else None
        regime_sl = ad.regime[test_start:end_inclusive] if getattr(ad, "regime", None) is not None else None
        sliced[sym] = AssetData(
            symbol=getattr(ad, "symbol", sym),
            open=ad.open[test_start:end_inclusive],
            high=ad.high[test_start:end_inclusive],
            low=ad.low[test_start:end_inclusive],
            close=ad.close[test_start:end_inclusive],
            score=ad.score[test_start:end_inclusive],
            atr=ad.atr[test_start:end_inclusive],
            tradeable=ad.tradeable[test_start:end_inclusive],
            first_valid_idx=first_valid_idx,
            cost_class=getattr(ad, "cost_class", "US_EQ_FROM_IN"),
            trend_score=trend_sc,
            regime=regime_sl,
        )
    return sliced

def run_backtest_on_split(
    date_index: pd.DatetimeIndex,
    assets: Any,
    config: Dict[str, Any],
    split: CPCVSplit,
    symbols: List[str],
    *,
    cost_free: bool = False,
) -> Dict[str, Any]:
    test_idx = split.test_idx
    if len(test_idx) == 0:
        return {}
    test_start = int(test_idx[0])
    test_end = int(test_idx[-1])
    start_ts = date_index[test_start]
    end_ts = date_index[test_end]
    start_dt = start_ts.to_pydatetime() if hasattr(start_ts, "to_pydatetime") else datetime.fromisoformat(str(start_ts))
    end_dt = end_ts.to_pydatetime() if hasattr(end_ts, "to_pydatetime") else datetime.fromisoformat(str(end_ts))

    if config.get("use_bot"):
        from validation.bot_backtest import run_bot_backtest

        sliced_index = date_index[test_start : test_end + 1]
        sliced_assets = _slice_assets_for_split(assets, test_start, test_end)
        return run_bot_backtest(sliced_index, sliced_assets, config)
    elif config.get("mode") == "allocation":
        from backtester.portfolio_simulator import AllocationEngine

        alloc_cfg = _config_to_alloc_config(config, symbols, start_dt, end_dt, cost_free=cost_free)
        engine = AllocationEngine(alloc_cfg)
        return engine.run(date_index, assets)
    else:
        sim_cfg = _config_to_sim_config(config, symbols, start_dt, end_dt, cost_free=cost_free)
        from backtester.portfolio_simulator import PortfolioSimulator

        sim = PortfolioSimulator(sim_cfg)
        return sim.run(date_index, assets)

def _sharpe_from_result(result: Dict[str, Any]) -> float:
    ec = result.get("equity_curve") or []
    if len(ec) < 2:
        return 0.0
    equities = [x.get("equity") for x in ec if x.get("equity") is not None]
    if len(equities) < 2:
        return 0.0
    arr = np.array(equities, dtype=float)
    ret = np.diff(arr) / (arr[:-1] + 1e-12)
    if ret.std() <= 0:
        return 0.0
    return float((ret.mean() / ret.std()) * np.sqrt(252))

def run_orchestrator(
    config_path: str,
    output_dir: Path,
    seed: int,
    *,
    data_loader: Optional[Callable[[Dict], Tuple[pd.DatetimeIndex, Any]]] = None,
) -> OrchestratorResult:
    """
    Run full pipeline: load config, prepare data, CPCV, backtest, GT-Score, DSR, persist.

    Parameters
    ----------
    config_path : str
        Path to YAML config (data, cpcv, mfbo, dsr_min, etc.).
    output_dir : Path
        Directory for winning_config.json, gt_scores, trials.
    seed : int
        Random seed for reproducibility.
    data_loader : callable, optional
        If set, (config_dict) -> (date_index, assets). Else uses prepare_multi_asset_data.

    Returns
    -------
    OrchestratorResult
    """
    import yaml

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    data_cfg = raw.get("data", {})
    cpcv_cfg = raw.get("cpcv", {})
    mfbo_cfg = raw.get("mfbo", {})
    dsr_min = float(raw.get("dsr_min", 0.05))
    cost_free = bool(raw.get("cost_free", False))
    gt_benchmark_scale = float(raw.get("gt_benchmark_scale", 1.0))
    symbols = data_cfg.get("symbols", ["SPY"])
    if isinstance(symbols, str):
        symbols = [s.strip() for s in symbols.split(",")]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if data_loader is not None:
        date_index, assets = data_loader(raw)
    else:
        start_s = data_cfg.get("start_date", "2018-01-01")
        end_s = data_cfg.get("end_date", "2024-01-01")
        start_dt = datetime.fromisoformat(start_s.replace("Z", "").split("T")[0])
        end_dt = datetime.fromisoformat(end_s.replace("Z", "").split("T")[0])
        date_index, assets = _load_data(
            symbols,
            start_dt,
            end_dt,
            data_cfg.get("asset_meta"),
        )

    n_samples = len(date_index)
    cpcv_config = CPCVConfig.from_dict(cpcv_cfg)
    splits = generate_cpcv_splits(cpcv_config, n_samples)
    if not splits:
        return OrchestratorResult(artifact_paths={"output_dir": str(out)})

    split_metadata = []
    for s in splits:
        test_idx = s.test_idx
        if len(test_idx) == 0:
            split_metadata.append({"split_id": s.split_id, "test_start_idx": None, "test_end_idx": None})
            continue
        test_start_idx = int(test_idx[0])
        test_end_idx = int(test_idx[-1])
        entry = {"split_id": s.split_id, "test_start_idx": test_start_idx, "test_end_idx": test_end_idx}
        try:
            start_ts = date_index[test_start_idx]
            end_ts = date_index[test_end_idx]
            entry["test_start_date"] = _ts_to_iso(start_ts)
            entry["test_end_date"] = _ts_to_iso(end_ts)
        except Exception:
            pass
        split_metadata.append(entry)
    with open(out / "cpcv_splits_metadata.json", "w") as f:
        json.dump(split_metadata, f, indent=2, default=str)

    mode = raw.get("mode", "tactical")
    is_allocation = mode == "allocation"

    search_space = mfbo_cfg.get("search_space", {})
    grid_search = raw.get("grid_search", {})
    locked_params = raw.get("locked", {})
    fidelity_spec = mfbo_cfg.get("fidelity_spec", {"low": {"cost": 1.0}, "high": {"cost": 5.0}})
    n_trials = int(mfbo_cfg.get("n_trials", 20))

    use_bot = bool(raw.get("use_bot", False))
    base_config: Dict[str, Any] = {
        "cost_free": cost_free,
        "initial_capital": float(raw.get("initial_capital", locked_params.get("initial_capital", 100_000.0))),
        "slippage_bps": float(locked_params.get("slippage_bps", raw.get("slippage_bps", 5.0))),
    }
    if is_allocation:
        base_config["mode"] = "allocation"
        for k, v in locked_params.items():
            base_config.setdefault(k, v)
    if use_bot:
        base_config["use_bot"] = True
        base_config.setdefault("kappa_tp", 1.5)
        base_config.setdefault("kappa_sl", 1.0)
        base_config.setdefault("T_max", 20)
        base_config.setdefault("min_position_notional", 3_000.0)

    all_configs: List[Dict[str, Any]] = []
    all_gt: List[Dict[str, Any]] = []
    all_sharpes_by_config: Dict[str, List[float]] = {}
    all_calmars_by_config: Dict[str, List[float]] = {}

    gt_turnover_lambda = float(raw.get("gt_turnover_lambda", 0.0))

    def _calmar_from_result(result: Dict[str, Any]) -> float:
        ec = result.get("equity_curve") or []
        if len(ec) < 2:
            return 0.0
        equities = [x.get("equity") for x in ec if x.get("equity") is not None]
        if len(equities) < 2 or equities[0] <= 0:
            return 0.0
        arr = np.array(equities, dtype=float)
        total_ret = arr[-1] / arr[0]
        n_years = len(arr) / 252.0
        if n_years <= 0 or total_ret <= 0:
            return 0.0
        cagr = total_ret ** (1.0 / n_years) - 1.0
        running_max = np.maximum.accumulate(arr)
        dd = (arr - running_max) / (running_max + 1e-12)
        max_dd = abs(float(dd.min()))
        if max_dd < 1e-8:
            return cagr * 100.0
        return cagr / max_dd

    def objective_fn(config: Dict[str, Any], fidelity: str) -> float:
        full_config = {**base_config, **config}
        gt_scores_split = []
        for s in splits:
            res = run_backtest_on_split(date_index, assets, full_config, s, symbols, cost_free=cost_free)
            if not res:
                continue
            strat, bench = equity_curve_from_result(res)
            if strat.empty or bench.empty:
                continue

            invested_frac = res.get("time_in_market_pct", 0.0) / 100.0 if res.get("time_in_market_pct") else None
            total_trades = res.get("total_trades", 0)
            n_days = max(len(strat), 1)
            turnover = total_trades / n_days

            gt = compute_gt_score(
                strat,
                bench,
                benchmark_scale=gt_benchmark_scale,
                invested_fraction=invested_frac,
                turnover_penalty_lambda=gt_turnover_lambda,
                turnover_rate=turnover,
            )
            gt_scores_split.append(gt)
            cfg_key = json.dumps(full_config, sort_keys=True)
            sharpe = _sharpe_from_result(res)
            calmar = _calmar_from_result(res)
            all_sharpes_by_config.setdefault(cfg_key, []).append(sharpe)
            all_calmars_by_config.setdefault(cfg_key, []).append(calmar)
        all_configs.append(full_config)
        all_gt.append({"config": full_config, "fidelity": fidelity, "gt_scores": gt_scores_split})

        if is_allocation:
            cfg_key = json.dumps(full_config, sort_keys=True)
            sharpes = all_sharpes_by_config.get(cfg_key, [])
            calmars = all_calmars_by_config.get(cfg_key, [])
            mean_sharpe = float(np.mean(sharpes)) if sharpes else 0.0
            mean_calmar = float(np.mean(calmars)) if calmars else 0.0
            return mean_calmar * max(mean_sharpe, 0.0)

        return float(np.mean(gt_scores_split)) if gt_scores_split else -1e6

    if grid_search and is_allocation:
        import itertools
        keys = list(grid_search.keys())
        values = [grid_search[k] for k in keys]
        grid_combos = list(itertools.product(*values))
        for combo in grid_combos:
            config = dict(zip(keys, combo))
            objective_fn(config, "high")
        trials = []
    else:
        try:
            from validation.mfbo import run_mfbo

            mfbo_result = run_mfbo(
                objective_fn,
                search_space,
                fidelity_spec,
                n_trials,
                seed,
                out,
            )
            trials = mfbo_result.get("trials", [])
        except RuntimeError:
            trials = []
            for i in range(n_trials):
                np.random.seed(seed + i)
                config = {}
                for k, v in search_space.items():
                    if isinstance(v, (list, tuple)) and len(v) > 0:
                        config[k] = v[np.random.randint(len(v))]
                objective_fn(config, "low")

    gt_matrix: List[Dict[str, Any]] = []
    for rec in all_gt:
        cfg_key = json.dumps(rec["config"], sort_keys=True)
        for split_id, gt in enumerate(rec.get("gt_scores", [])):
            gt_matrix.append({"config_key": cfg_key, "split_id": split_id, "gt_score": gt})

    dsr_by_config: Dict[str, float] = {}
    n_trials_used = max(len(all_configs), 1)
    for cfg_key, sharpes in all_sharpes_by_config.items():
        if sharpes:
            dsr_by_config[cfg_key] = compute_dsr(sharpes, n_trials_used)

    best_config = None
    best_dsr = None
    best_mean_gt = None
    best_score = -np.inf
    dsr_passed = False

    if is_allocation:
        for cfg_key in set(r["config_key"] for r in gt_matrix):
            sharpes = all_sharpes_by_config.get(cfg_key, [])
            calmars = all_calmars_by_config.get(cfg_key, [])
            mean_sharpe = float(np.mean(sharpes)) if sharpes else 0.0
            mean_calmar = float(np.mean(calmars)) if calmars else 0.0
            composite = mean_calmar * max(mean_sharpe, 0.0)
            if composite > best_score:
                best_score = composite
                gts = [r["gt_score"] for r in gt_matrix if r["config_key"] == cfg_key]
                best_mean_gt = float(np.mean(gts)) if gts else None
                best_dsr = dsr_by_config.get(cfg_key)
                for c in all_configs:
                    if json.dumps(c, sort_keys=True) == cfg_key:
                        best_config = c
                        break
        dsr_passed = True
    else:
        for cfg_key, dsr in dsr_by_config.items():
            if dsr < dsr_min:
                continue
            dsr_passed = True
            gts = [r["gt_score"] for r in gt_matrix if r["config_key"] == cfg_key]
            mean_gt = float(np.mean(gts)) if gts else -1e6
            if best_mean_gt is None or mean_gt > best_mean_gt:
                best_mean_gt = mean_gt
                best_dsr = dsr
                for c in all_configs:
                    if json.dumps(c, sort_keys=True) == cfg_key:
                        best_config = c
                        break

        if not dsr_passed and dsr_by_config:
            for cfg_key, dsr in dsr_by_config.items():
                gts = [r["gt_score"] for r in gt_matrix if r["config_key"] == cfg_key]
                mean_gt = float(np.mean(gts)) if gts else -1e6
                if best_mean_gt is None or mean_gt > best_mean_gt:
                    best_mean_gt = mean_gt
                    best_dsr = dsr
                    for c in all_configs:
                        if json.dumps(c, sort_keys=True) == cfg_key:
                            best_config = c
                            break

    winning_path = out / "winning_config.json"
    with open(winning_path, "w") as f:
        json.dump(
            {
                "config": best_config,
                "dsr": best_dsr,
                "mean_gt_score": best_mean_gt,
                "n_trials": n_trials_used,
                "dsr_passed": dsr_passed,
            },
            f,
            indent=2,
            default=str,
        )

    gt_path = out / "gt_scores.json"
    with open(gt_path, "w") as f:
        json.dump(gt_matrix, f, indent=2, default=str)

    try:
        import fastparquet
        df_gt = pd.DataFrame(gt_matrix)
        fastparquet.write(out / "gt_scores.parquet", df_gt, index=False)
    except Exception:
        pass

    return OrchestratorResult(
        winning_config=best_config,
        winning_dsr=best_dsr,
        winning_mean_gt=best_mean_gt,
        gt_scores={"matrix": gt_matrix, "dsr_by_config": dsr_by_config},
        dsr_by_config=dsr_by_config,
        n_trials=n_trials_used,
        artifact_paths={
            "winning_config": str(winning_path),
            "gt_scores": str(gt_path),
            "output_dir": str(out),
        },
    )
