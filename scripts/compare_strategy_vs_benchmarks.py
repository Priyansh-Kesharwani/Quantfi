#!/usr/bin/env python3
"""
Compare strategy return vs buy-and-hold and uniform periodic benchmarks.

Reads sim_summary.json (must contain benchmarks from run_sim_diagnostics with run_benchmarks=True)
and writes a short comparison to validation/debug/strategy_vs_benchmarks.md.
If summary path has no benchmarks, run run_sim_diagnostics first with default run_benchmarks.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description="Compare strategy vs B&H and uniform periodic benchmarks")
    ap.add_argument("--summary", type=str, default="validation/debug/sim_summary.json", help="Path to sim_summary.json")
    ap.add_argument("--out", type=str, default="validation/debug/strategy_vs_benchmarks.md", help="Output markdown path")
    args = ap.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.is_absolute():
        summary_path = PROJECT_ROOT / summary_path
    if not summary_path.exists():
        print(f"Summary not found: {summary_path}. Run: python3 scripts/run_sim_diagnostics.py --symbols AAPL,SPY --out validation/debug", file=sys.stderr)
        return 1

    with open(summary_path) as f:
        summary = json.load(f)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    strat_ret = summary.get("total_return_pct")
    benchmarks = summary.get("benchmarks") or {}
    bnh = benchmarks.get("buy_and_hold") or {}
    unif = benchmarks.get("uniform_periodic") or {}
    bnh_ret = bnh.get("total_return_pct")
    unif_ret = unif.get("total_return_pct")

    lines = []
    lines.append("# Strategy vs Benchmarks (Long-Only Passive)")
    lines.append("")
    lines.append("Market timing (signal-based strategy) vs buy-and-hold and uniform periodic investment.")
    lines.append("")
    lines.append("| Series | Total return % | CAGR % | Max DD % |")
    lines.append("|--------|-----------------|--------|----------|")
    def row(name, ret, cagr=None, dd=None):
        r = f"{ret:.2f}" if ret is not None else "—"
        c = f"{cagr:.2f}" if cagr is not None else "—"
        d = f"{dd:.2f}" if dd is not None else "—"
        return f"| {name} | {r} | {c} | {d} |"
    lines.append(row("Strategy", strat_ret, summary.get("cagr_pct"), summary.get("max_drawdown_pct")))
    lines.append(row("Buy & hold", bnh_ret, bnh.get("cagr_pct"), bnh.get("max_drawdown_pct")))
    lines.append(row("Uniform periodic", unif_ret, unif.get("cagr_pct"), None))
    lines.append("")
    if strat_ret is not None and bnh_ret is not None:
        diff_bnh = strat_ret - bnh_ret
        lines.append(f"**Strategy vs B&H:** {diff_bnh:+.2f} pp.")
    if strat_ret is not None and unif_ret is not None:
        diff_unif = strat_ret - unif_ret
        lines.append(f"**Strategy vs uniform periodic:** {diff_unif:+.2f} pp.")
    lines.append("")
    lines.append("(Generated from " + str(summary_path.name) + ")")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Comparison written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
