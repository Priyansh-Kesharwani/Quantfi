#!/usr/bin/env python3
"""
Suggest entry score threshold from decile forward returns (ic_results.json).

Picks a horizon and suggests entering when score is in deciles with positive mean
forward return (e.g. decile >= 8). Outputs a recommended threshold or decile->return table.
Score boundaries are approximate (decile 8 ≈ top 20% of scores).
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    ap = argparse.ArgumentParser(description="Suggest entry threshold from decile forward returns")
    ap.add_argument("--input", type=str, default="validation/debug/ic_results.json", help="Path to ic_results.json")
    ap.add_argument("--horizon", type=int, default=20, help="Forward horizon (bars) to use for suggestion")
    ap.add_argument("--min-decile", type=int, default=8, help="Suggest entering when score in decile >= this")
    ap.add_argument("--asset", type=str, default=None, help="Asset to use (default: first in file)")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = PROJECT_ROOT / in_path
    if not in_path.exists():
        print(f"Input not found: {in_path}", file=sys.stderr)
        return 1

    with open(in_path) as f:
        data = json.load(f)

    asset = args.asset
    if not asset and data:
        asset = next((k for k in data if isinstance(data[k], dict) and "error" not in data[k]), None)
    if not asset or asset not in data or not isinstance(data[asset], dict) or "decile_forward_returns" not in data[asset]:
        print("No valid asset or decile data.", file=sys.stderr)
        return 1

    blob = data[asset]
    deciles_by_h = blob.get("decile_forward_returns", {})
    h_key = f"horizon_{args.horizon}"
    if h_key not in deciles_by_h:
        for k, v in deciles_by_h.items():
            if isinstance(v, dict) and v.get("horizon") == args.horizon:
                h_key = k
                break
    if h_key not in deciles_by_h:
        print(f"Horizon {args.horizon} not found. Available: {list(deciles_by_h.keys())}", file=sys.stderr)
        return 1

    deciles = deciles_by_h[h_key].get("deciles", [])
    if not deciles:
        print("No decile data.", file=sys.stderr)
        return 1

    min_d = args.min_decile
    print(f"Asset: {asset}  Horizon: {args.horizon}d  Min decile for entry: {min_d}")
    print("")
    print("Decile | Mean forward return | Std error | n_bars")
    print("-------|---------------------|----------|-------")
    for d in deciles:
        q = d.get("decile")
        mean_ret = d.get("mean_forward_return")
        se = d.get("std_error")
        n = d.get("n_bars", 0)
        if mean_ret is None:
            continue
        pct = mean_ret * 100
        sep = se * 100 if se is not None else 0
        rec = "  <-- suggest entry in this range" if q >= min_d and mean_ret > 0 else ""
        print(f"   {q:2d}   | {pct:7.2f}%             | {sep:6.2f}%   | {n:5d}{rec}")

    suggested_score = 50 + (min_d - 1) * 5
    suggested_score = max(50, min(95, 50 + (10 - min_d) * 5))
    print("")
    print(f"Suggested entry_score_threshold (approximate): {suggested_score}")
    print("(Use with --min-decile to tune; then run turnover_sensitivity around this value.)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
