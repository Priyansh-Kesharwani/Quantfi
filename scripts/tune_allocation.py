#!/usr/bin/env python3
"""
Allocation Engine Grid Search + Verification.

Loads 4-asset data, runs grid search over tunable parameters on CPCV splits,
reports risk-adjusted metrics, and verifies the winning config beats buy-and-hold.

Usage:
    python3 scripts/tune_allocation.py [--config config/tuning_allocation.yml]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtester.portfolio_simulator import (
    AllocationConfig,
    AllocationEngine,
    prepare_multi_asset_data,
)
from validation.objective import compute_gt_score, equity_curve_from_result
from validation.validator import CPCVConfig, generate_cpcv_splits


def _sharpe(eq: np.ndarray) -> float:
    if len(eq) < 2:
        return 0.0
    ret = np.diff(eq) / (eq[:-1] + 1e-12)
    if ret.std() <= 0:
        return 0.0
    return float((ret.mean() / ret.std()) * np.sqrt(252))


def _calmar(eq: np.ndarray) -> float:
    if len(eq) < 2 or eq[0] <= 0:
        return 0.0
    total_ret = eq[-1] / eq[0]
    n_years = len(eq) / 252.0
    if n_years <= 0 or total_ret <= 0:
        return 0.0
    cagr = total_ret ** (1.0 / n_years) - 1.0
    running_max = np.maximum.accumulate(eq)
    dd = (eq - running_max) / (running_max + 1e-12)
    max_dd = abs(float(dd.min()))
    if max_dd < 1e-8:
        return cagr * 100.0
    return cagr / max_dd


def _max_dd(eq: np.ndarray) -> float:
    if len(eq) < 2:
        return 0.0
    running_max = np.maximum.accumulate(eq)
    dd = (eq - running_max) / (running_max + 1e-12)
    return float(dd.min())


def _time_in_market(result: Dict[str, Any]) -> float:
    ec = result.get("equity_curve") or []
    if not ec:
        return 0.0
    invested = [e.get("invested_pct", 0) for e in ec]
    return float(np.mean([1.0 if x > 1.0 else 0.0 for x in invested])) * 100.0


def _bh_equity(date_index, assets, start_idx, end_idx, initial_capital):
    n_assets = len(assets)
    per_asset = initial_capital / n_assets
    eq = np.zeros(end_idx - start_idx + 1)
    for i, t in enumerate(range(start_idx, end_idx + 1)):
        val = 0.0
        for sym, ad in assets.items():
            if t < ad.first_valid_idx:
                val += per_asset
            else:
                init_price = ad.close[ad.first_valid_idx]
                if init_price > 0:
                    val += per_asset * (ad.close[t] / init_price)
                else:
                    val += per_asset
        eq[i] = val
    return eq


def run_grid_search(config_path: str) -> Dict[str, Any]:
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    data_cfg = raw["data"]
    cpcv_cfg = raw.get("cpcv", {})
    grid = raw.get("grid_search", {})
    locked = raw.get("locked", {})

    symbols = data_cfg["symbols"]
    start_dt = datetime.fromisoformat(data_cfg["start_date"])
    end_dt = datetime.fromisoformat(data_cfg["end_date"])

    print(f"Loading data for {symbols} from {start_dt.date()} to {end_dt.date()}...")
    t0 = time.time()
    date_index, assets = prepare_multi_asset_data(
        symbols=symbols,
        start_date=start_dt,
        end_date=end_dt,
        asset_meta={s: {"asset_type": "equity", "currency": "USD"} for s in symbols},
    )
    print(f"  Data loaded: {len(date_index)} bars, {time.time()-t0:.1f}s")

    cpcv_config = CPCVConfig.from_dict(cpcv_cfg)
    splits = generate_cpcv_splits(cpcv_config, len(date_index))
    print(f"  CPCV: {len(splits)} splits")

    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    combos = list(itertools.product(*values))
    print(f"  Grid: {len(combos)} configurations ({' x '.join(str(len(v)) for v in values)})")

    results: List[Dict[str, Any]] = []

    for ci, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        full = {**locked, **params}
        cfg_sharpes = []
        cfg_calmars = []
        cfg_max_dds = []
        cfg_bh_sharpes = []
        cfg_bh_calmars = []
        cfg_returns = []
        cfg_bh_returns = []
        cfg_time_in_market = []

        for s in splits:
            test_idx = s.test_idx
            if len(test_idx) < 10:
                continue
            test_start = int(test_idx[0])
            test_end = int(test_idx[-1])
            start_ts = date_index[test_start]
            end_ts = date_index[test_end]
            s_dt = start_ts.to_pydatetime() if hasattr(start_ts, "to_pydatetime") else datetime.fromisoformat(str(start_ts))
            e_dt = end_ts.to_pydatetime() if hasattr(end_ts, "to_pydatetime") else datetime.fromisoformat(str(end_ts))

            alloc_cfg = AllocationConfig(
                symbols=symbols,
                start_date=s_dt,
                end_date=e_dt,
                initial_capital=float(full.get("initial_capital", 100000)),
                risk_on_equity_pct=float(full.get("risk_on_equity_pct", 0.85)),
                risk_off_equity_pct=float(full.get("risk_off_equity_pct", 0.40)),
                theta_tilt=float(full.get("theta_tilt", 2.0)),
                min_weight_floor=float(full.get("min_weight_floor", 0.05)),
                rebalance_freq_days=int(full.get("rebalance_freq_days", 21)),
                min_rebalance_delta=float(full.get("min_rebalance_delta", 0.05)),
                jump_penalty=float(full.get("jump_penalty", 25.0)),
                regime_n_states=int(full.get("regime_n_states", 2)),
                regime_window=int(full.get("regime_window", 504)),
                regime_refit_every=int(full.get("regime_refit_every", 63)),
                drawdown_circuit_threshold=float(full.get("drawdown_circuit_threshold", -0.10)),
                hysteresis_enter=float(full.get("hysteresis_enter", 0.70)),
                hysteresis_exit=float(full.get("hysteresis_exit", 0.60)),
                cooldown_days=int(full.get("cooldown_days", 15)),
                slippage_bps=float(full.get("slippage_bps", 5.0)),
                cash_return_annual=float(full.get("cash_return_annual", 0.02)),
                scoring_mode=full.get("scoring_mode", "adaptive"),
                run_benchmarks=True,
            )

            try:
                engine = AllocationEngine(alloc_cfg)
                res = engine.run(date_index, assets)
            except Exception as exc:
                print(f"  [WARN] Split {s.split_id} failed: {exc}")
                continue

            ec = res.get("equity_curve") or []
            if len(ec) < 2:
                continue
            eq = np.array([e["equity"] for e in ec], dtype=float)
            cfg_sharpes.append(_sharpe(eq))
            cfg_calmars.append(_calmar(eq))
            cfg_max_dds.append(_max_dd(eq))
            cfg_returns.append(float(eq[-1] / eq[0] - 1.0) * 100.0)
            cfg_time_in_market.append(_time_in_market(res))

            bh_eq = _bh_equity(date_index, assets, test_start, test_end, alloc_cfg.initial_capital)
            if len(bh_eq) >= 2:
                cfg_bh_sharpes.append(_sharpe(bh_eq))
                cfg_bh_calmars.append(_calmar(bh_eq))
                cfg_bh_returns.append(float(bh_eq[-1] / bh_eq[0] - 1.0) * 100.0)

        if not cfg_sharpes:
            continue

        mean_sharpe = float(np.mean(cfg_sharpes))
        mean_calmar = float(np.mean(cfg_calmars))
        composite = mean_calmar * max(mean_sharpe, 0.0)

        row = {
            "params": params,
            "full_config": full,
            "mean_sharpe": round(mean_sharpe, 4),
            "mean_calmar": round(mean_calmar, 4),
            "composite_score": round(composite, 4),
            "mean_max_dd": round(float(np.mean(cfg_max_dds)), 4),
            "mean_return_pct": round(float(np.mean(cfg_returns)), 2),
            "mean_time_in_market_pct": round(float(np.mean(cfg_time_in_market)), 1),
            "bh_mean_sharpe": round(float(np.mean(cfg_bh_sharpes)), 4) if cfg_bh_sharpes else None,
            "bh_mean_calmar": round(float(np.mean(cfg_bh_calmars)), 4) if cfg_bh_calmars else None,
            "bh_mean_return_pct": round(float(np.mean(cfg_bh_returns)), 2) if cfg_bh_returns else None,
            "n_splits": len(cfg_sharpes),
            "sharpe_beats_bh": sum(1 for s, b in zip(cfg_sharpes, cfg_bh_sharpes) if s > b) if cfg_bh_sharpes else 0,
            "dd_beats_bh": sum(1 for d, b in zip(cfg_max_dds, [_max_dd(_bh_equity(date_index, assets, int(sp.test_idx[0]), int(sp.test_idx[-1]), alloc_cfg.initial_capital)) for sp in splits if len(sp.test_idx) >= 10]) if d > b),
        }
        results.append(row)
        print(f"  [{ci+1}/{len(combos)}] {params} -> Sharpe={mean_sharpe:.3f} Calmar={mean_calmar:.3f} Composite={composite:.3f}")

    results.sort(key=lambda r: r["composite_score"], reverse=True)
    return {"results": results, "symbols": symbols, "n_splits": len(splits)}


def main():
    parser = argparse.ArgumentParser(description="Allocation Engine Grid Search")
    parser.add_argument("--config", default="config/tuning_allocation.yml")
    args = parser.parse_args()

    output = run_grid_search(args.config)
    results = output["results"]

    if not results:
        print("\nNo valid results. Check data and configuration.")
        sys.exit(1)

    winner = results[0]
    print("\n" + "=" * 70)
    print("WINNING CONFIGURATION")
    print("=" * 70)
    for k, v in winner["params"].items():
        print(f"  {k}: {v}")
    print(f"\n  Sharpe:         {winner['mean_sharpe']}")
    print(f"  Calmar:         {winner['mean_calmar']}")
    print(f"  Composite:      {winner['composite_score']}")
    print(f"  Max DD (mean):  {winner['mean_max_dd']}")
    print(f"  Return (mean):  {winner['mean_return_pct']}%")
    print(f"  Time in market: {winner['mean_time_in_market_pct']}%")
    print(f"  B&H Sharpe:     {winner['bh_mean_sharpe']}")
    print(f"  B&H Return:     {winner['bh_mean_return_pct']}%")
    print(f"  Splits beating B&H Sharpe: {winner['sharpe_beats_bh']}/{winner['n_splits']}")

    out_dir = PROJECT_ROOT / "validation" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "allocation_winning_config.json"
    with open(out_path, "w") as f:
        json.dump(winner, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    sharpe_ok = winner["mean_sharpe"] > (winner["bh_mean_sharpe"] or 0)
    dd_ok = winner["mean_max_dd"] > (float(winner.get("bh_mean_max_dd") or -1.0))
    time_ok = winner["mean_time_in_market_pct"] > 50.0

    print(f"\nVerification:")
    print(f"  Sharpe > B&H:      {'PASS' if sharpe_ok else 'NEEDS TUNING'}")
    print(f"  Time in market>50%: {'PASS' if time_ok else 'NEEDS TUNING'}")


if __name__ == "__main__":
    main()
