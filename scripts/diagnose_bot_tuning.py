#!/usr/bin/env python3
"""Run one CPCV split with bot backtest and print why GT-Score might be -1e6."""
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
backend = root / "backend"
if str(backend) not in sys.path:
    sys.path.insert(0, str(backend))
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import yaml
from datetime import datetime

from validation.orchestrator import _load_data, run_backtest_on_split, _slice_assets_for_split
from validation.validator import CPCVConfig, generate_cpcv_splits
from validation.objective import equity_curve_from_result, compute_gt_score

def main():
    config_path = root / "config" / "tuning_cpcv_bot.yml"
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}
    data_cfg = raw.get("data", {})
    symbols = data_cfg.get("symbols", ["SPY", "AAPL"])
    if isinstance(symbols, str):
        symbols = [s.strip() for s in symbols.split(",")]
    start_dt = datetime.fromisoformat(data_cfg.get("start_date", "2018-01-01").split("T")[0])
    end_dt = datetime.fromisoformat(data_cfg.get("end_date", "2024-01-01").split("T")[0])

    print("Loading data...")
    date_index, assets = _load_data(symbols, start_dt, end_dt, data_cfg.get("asset_meta"))
    n_samples = len(date_index)
    print(f"  date_index len={n_samples}, assets={list(assets.keys())}")

    cpcv_cfg = raw.get("cpcv", {})
    config = CPCVConfig.from_dict(cpcv_cfg)
    splits = generate_cpcv_splits(config, n_samples)
    print(f"  splits={len(splits)}")

    use_bot = True
    base_config = {
        "use_bot": True,
        "entry_score_threshold": 70,
        "max_positions": 10,
        "kappa_tp": 1.5,
        "kappa_sl": 1.0,
        "T_max": 20,
        "initial_capital": 100_000.0,
        "slippage_bps": 5.0,
        "cost_free": True,
    }

    for i, s in enumerate(splits):
        test_idx = s.test_idx
        if len(test_idx) == 0:
            print(f"Split {i}: empty test_idx")
            continue
        test_start = int(test_idx[0])
        test_end = int(test_idx[-1])
        slice_len = test_end - test_start + 1
        print(f"\nSplit {i}: test_start={test_start}, test_end={test_end}, slice_len={slice_len}")

        res = run_backtest_on_split(date_index, assets, base_config, s, symbols, cost_free=False)
        if not res:
            print("  res is empty")
            continue
        ec = res.get("equity_curve") or []
        bnh = (res.get("benchmarks") or {}).get("buy_and_hold") or {}
        bnh_curve = bnh.get("equity_curve") or []
        print(f"  equity_curve len={len(ec)}, bnh_curve len={len(bnh_curve)}")
        if ec:
            dates = [x.get("date") for x in ec if x.get("date")]
            equity = [x.get("equity") for x in ec if x.get("equity") is not None]
            print(f"  dates len={len(dates)}, equity len={len(equity)}, match={len(dates)==len(equity)}")
            if dates and equity:
                print(f"  first date={dates[0]}, first equity={equity[0]}")
                print(f"  last date={dates[-1]}, last equity={equity[-1]}")

        strat, bench = equity_curve_from_result(res)
        print(f"  strategy len={len(strat)}, empty={strat.empty}")
        print(f"  benchmark len={len(bench)}, empty={bench.empty}")

        if strat.empty or bench.empty:
            print("  -> GT-Score skipped (empty strat or bench)")
            continue
        gt = compute_gt_score(strat, bench)
        print(f"  GT-Score={gt}")
        if gt <= -1e5:
            eq = strat.dropna()
            bm = bench.reindex(eq.index).ffill().bfill()
            print(f"  eq len={len(eq)}, bm.notna().sum()={bm.notna().sum()}")
            if len(eq) >= 2:
                ret = eq.pct_change().dropna()
                ret_bm = bm.pct_change().dropna()
                common = ret.index.intersection(ret_bm.index)
                if len(common) >= 2:
                    mu = float(ret.loc[common].mean())
                    mu_bm = float(ret_bm.loc[common].mean())
                    print(f"  mu(strat)={mu:.6f}, mu(bench)={mu_bm:.6f}, underperform={mu <= mu_bm}")
    print("\nDone.")

if __name__ == "__main__":
    main()
