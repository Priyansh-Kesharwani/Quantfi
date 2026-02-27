#!/usr/bin/env python3
"""
Noise sensitivity: run portfolio sim with and without score noise, compare return and trade count.

Uses SimConfig.score_noise_sigma: when > 0, score at each bar gets Gaussian noise (clamped 0-100).
Output: validation/debug/noise_sensitivity.md and optional JSON.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

def main():
    ap = argparse.ArgumentParser(description="Noise sensitivity: baseline vs noisy scores")
    ap.add_argument("--symbols", type=str, default="AAPL,SPY", help="Comma-separated symbols")
    ap.add_argument("--start", type=str, default="2020-01-01")
    ap.add_argument("--end", type=str, default="2023-12-31")
    ap.add_argument("--sigma", type=str, default="0,2,5", help="Comma-separated score_noise_sigma values")
    ap.add_argument("--out-dir", type=str, default="validation/debug")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    args = ap.parse_args()

    seed = args.seed

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    sigmas = [float(x.strip()) for x in args.sigma.split(",") if x.strip()]
    if not sigmas:
        sigmas = [0.0, 2.0, 5.0]
    start_dt = datetime.fromisoformat(args.start)
    end_dt = datetime.fromisoformat(args.end)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_path = PROJECT_ROOT / "backtester" / "portfolio_simulator.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("portfolio_simulator", str(sim_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["portfolio_simulator"] = mod
    spec.loader.exec_module(mod)

    asset_meta = {sym: {"asset_type": "equity", "currency": "USD"} for sym in symbols}
    print(f"Preparing data for {symbols} ...", file=sys.stderr)
    date_index, assets_data = mod.prepare_multi_asset_data(
        symbols=symbols,
        start_date=start_dt,
        end_date=end_dt,
        asset_meta=asset_meta,
        usd_inr_rate=83.5,
    )

    rows = []
    for sigma in sigmas:
        np.random.seed(seed)
        config = mod.SimConfig(
            symbols=symbols,
            start_date=start_dt,
            end_date=end_dt,
            initial_capital=100_000.0,
            entry_score_threshold=70.0,
            slippage_bps=5.0,
            cost_free=False,
            run_benchmarks=False,
            score_noise_sigma=sigma,
        )
        sim = mod.PortfolioSimulator(config)
        result = sim.run(date_index, assets_data)
        ret = result.get("total_return_pct")
        n_trades = result.get("total_trades", 0)
        rows.append({"score_noise_sigma": sigma, "total_return_pct": ret, "total_trades": n_trades})
        print(f"  sigma={sigma} -> return={ret}% trades={n_trades}", file=sys.stderr)

    baseline = next((r for r in rows if r["score_noise_sigma"] == 0), rows[0])
    lines = []
    lines.append("# Noise Sensitivity")
    lines.append("")
    lines.append("Impact of adding Gaussian noise to Entry/Exit scores (stability test).")
    lines.append("")
    lines.append("| score_noise_sigma | total_return_pct | total_trades |")
    lines.append("|-------------------|-------------------|--------------|")
    for r in rows:
        lines.append(f"| {r['score_noise_sigma']} | {r['total_return_pct']} | {r['total_trades']} |")
    lines.append("")
    for r in rows:
        if r["score_noise_sigma"] != 0 and baseline.get("total_return_pct") is not None and r.get("total_return_pct") is not None:
            diff = r["total_return_pct"] - baseline["total_return_pct"]
            lines.append(f"- sigma={r['score_noise_sigma']}: return change vs baseline = {diff:+.2f} pp, trades = {r['total_trades']}")
    lines.append("")

    md_path = out_dir / "noise_sensitivity.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report written to {md_path}")

    json_path = out_dir / "noise_sensitivity.json"
    with open(json_path, "w") as f:
        json.dump({"rows": rows, "seed": seed}, f, indent=2)
    return 0

if __name__ == "__main__":
    sys.exit(main())
