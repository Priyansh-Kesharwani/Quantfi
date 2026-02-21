#!/usr/bin/env python3
"""
Run portfolio simulation from CLI for diagnostics (cost-free baseline, trade export).

Usage:
  python scripts/run_sim_diagnostics.py --symbols AAPL,SPY --start 2020-01-01 --end 2023-12-31 --out /tmp/out
  python scripts/run_sim_diagnostics.py --symbols AAPL,SPY --start 2020-01-01 --end 2023-12-31 --cost-free --out /tmp/out
  python scripts/run_sim_diagnostics.py --config config/phase3.yml --cost-free --out validation/debug/sim_summary.json

Requires: backend and project root on PYTHONPATH so data_providers and backtester resolve.
Run from project root: python scripts/run_sim_diagnostics.py ...
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Project root and backend on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def main():
    ap = argparse.ArgumentParser(description="Run portfolio sim for diagnostics (cost-free, trade export)")
    ap.add_argument("--config", type=str, default=None, help="Path to phase3.yml (optional; used for assets if --symbols not set)")
    ap.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols, e.g. AAPL,SPY,GLD")
    ap.add_argument("--start", type=str, default="2020-01-01", help="Start date YYYY-MM-DD")
    ap.add_argument("--end", type=str, default="2023-12-31", help="End date YYYY-MM-DD")
    ap.add_argument("--cost-free", action="store_true", help="Zero costs and zero slippage (diagnostic baseline)")
    ap.add_argument("--export-trades", action="store_true", default=True, help="Write trades to CSV under --out dir")
    ap.add_argument("--out", type=str, default="validation/debug", help="Output dir for summary JSON and trades CSV")
    args = ap.parse_args()

    # Resolve symbols
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
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load simulator module (uses backend data_providers when prepare_multi_asset_data runs)
    sim_path = PROJECT_ROOT / "backtester" / "portfolio_simulator.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("portfolio_simulator", str(sim_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["portfolio_simulator"] = mod
    spec.loader.exec_module(mod)

    asset_meta = {sym: {"asset_type": "equity", "currency": "USD"} for sym in symbols}
    usd_inr = 83.5

    print(f"Preparing data for {symbols} from {start_dt.date()} to {end_dt.date()} (cost_free={args.cost_free}) ...")
    date_index, assets_data = mod.prepare_multi_asset_data(
        symbols=symbols,
        start_date=start_dt,
        end_date=end_dt,
        asset_meta=asset_meta,
        usd_inr_rate=usd_inr,
    )

    config = mod.SimConfig(
        symbols=symbols,
        start_date=start_dt,
        end_date=end_dt,
        initial_capital=100_000.0,
        entry_score_threshold=70.0,
        slippage_bps=0.0 if args.cost_free else 5.0,
        cost_free=args.cost_free,
        run_benchmarks=True,
    )

    simulator = mod.PortfolioSimulator(config)
    result = simulator.run(date_index, assets_data)

    # Summary JSON (include benchmarks when run_benchmarks=True)
    summary = {
        "cost_free": args.cost_free,
        "symbols": symbols,
        "start": args.start,
        "end": args.end,
        "total_return_pct": result.get("total_return_pct"),
        "total_trades": result.get("total_trades"),
        "total_costs": result.get("total_costs"),
        "cost_drag_pct": result.get("cost_drag_pct"),
        "max_drawdown_pct": result.get("max_drawdown_pct"),
        "cagr_pct": result.get("cagr_pct"),
        "benchmarks": result.get("benchmarks"),
    }
    summary_path = out_dir / "sim_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {summary_path}")

    # Trades CSV
    if args.export_trades and result.get("trades"):
        run_id = "cli_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = out_dir / f"trades_{run_id}.csv"
        mod.export_trades_csv(str(csv_path), trades=result["trades"], run_id=run_id)
        print(f"Trades written to {csv_path}")

    print(f"Total return: {result.get('total_return_pct')}% | Trades: {result.get('total_trades')} | Costs: {result.get('total_costs')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
