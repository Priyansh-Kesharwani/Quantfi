#!/usr/bin/env python3
"""
Turnover sensitivity: sweep entry_score_threshold from 50 to 90; record total_trades
and total_return_pct (and optionally win_rate, cost_drag_pct) per run.

Usage:
  python scripts/turnover_sensitivity.py --symbols AAPL,SPY --start 2020-01-01 --end 2023-12-31
  python scripts/turnover_sensitivity.py --config config/phase3.yml --out validation/debug/turnover_sensitivity.csv

Run from project root. Same path/import setup as run_sim_diagnostics (backend + project root).
"""

import argparse
import csv
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

def main():
    ap = argparse.ArgumentParser(description="Turnover sensitivity: sweep entry threshold 50–90")
    ap.add_argument("--config", type=str, default=None, help="Path to phase3.yml (optional; for assets)")
    ap.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols")
    ap.add_argument("--start", type=str, default="2020-01-01", help="Start date YYYY-MM-DD")
    ap.add_argument("--end", type=str, default="2023-12-31", help="End date YYYY-MM-DD")
    ap.add_argument("--out", type=str, default="validation/debug/turnover_sensitivity.csv", help="Output CSV path")
    args = ap.parse_args()

    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols and args.config:
        import yaml
        cfg_path = PROJECT_ROOT / args.config
        if cfg_path.exists():
            with open(cfg_path) as f:
                data = yaml.safe_load(f) or {}
            assets = data.get("data", {}).get("assets", [])
            symbols = [a["symbol"] for a in assets if isinstance(a, dict) and a.get("symbol")]
    if not symbols:
        symbols = ["AAPL", "SPY"]
        print(f"No symbols from --symbols or --config; using {symbols}", file=sys.stderr)

    start_dt = datetime.fromisoformat(args.start)
    end_dt = datetime.fromisoformat(args.end)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sim_path = PROJECT_ROOT / "backtester" / "portfolio_simulator.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("portfolio_simulator", str(sim_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["portfolio_simulator"] = mod
    spec.loader.exec_module(mod)

    asset_meta = {sym: {"asset_type": "equity", "currency": "USD"} for sym in symbols}
    usd_inr = 83.5

    print(f"Preparing data for {symbols} from {start_dt.date()} to {end_dt.date()} ...")
    date_index, assets_data = mod.prepare_multi_asset_data(
        symbols=symbols,
        start_date=start_dt,
        end_date=end_dt,
        asset_meta=asset_meta,
        usd_inr_rate=usd_inr,
    )

    thresholds = [50, 55, 60, 65, 70, 75, 80, 85, 90]
    rows = []

    for thresh in thresholds:
        config = mod.SimConfig(
            symbols=symbols,
            start_date=start_dt,
            end_date=end_dt,
            initial_capital=100_000.0,
            entry_score_threshold=float(thresh),
            slippage_bps=5.0,
            cost_free=False,
            run_benchmarks=False,
        )
        sim = mod.PortfolioSimulator(config)
        result = sim.run(date_index, assets_data)
        rows.append({
            "entry_score_threshold": thresh,
            "total_trades": result.get("total_trades", 0),
            "total_return_pct": result.get("total_return_pct"),
            "win_rate": result.get("win_rate"),
            "cost_drag_pct": result.get("cost_drag_pct"),
        })
        print(f"  threshold={thresh} -> trades={result.get('total_trades')} return_pct={result.get('total_return_pct')}")

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["entry_score_threshold", "total_trades", "total_return_pct", "win_rate", "cost_drag_pct"])
        w.writeheader()
        w.writerows(rows)

    print(f"Written {out_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
