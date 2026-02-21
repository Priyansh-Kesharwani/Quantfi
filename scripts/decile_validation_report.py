#!/usr/bin/env python3
"""
Decile validation report: read ic_results.json and identify where Entry_Score
returns fall off (flat or negative deciles vs strong upper deciles).

Output: validation/debug/decile_validation_report.md and optional JSON summary.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description="Generate decile validation report from ic_results.json")
    ap.add_argument("--input", type=str, default="validation/debug/ic_results.json", help="Path to ic_results.json")
    ap.add_argument("--out-dir", type=str, default="validation/debug", help="Output directory for report")
    ap.add_argument("--flat-threshold", type=float, default=0.001, help="Mean fwd return below this (abs) is 'flat'")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = PROJECT_ROOT / in_path
    if not in_path.exists():
        print(f"Input not found: {in_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(in_path) as f:
        data = json.load(f)

    flat_thresh = args.flat_threshold
    lines = []
    lines.append("# Decile Validation Report")
    lines.append("")
    lines.append("Identifies deciles where mean forward return is flat or negative (Entry_Score range to avoid or downweight).")
    lines.append("")
    json_summary = {}

    for asset, blob in data.items():
        if not isinstance(blob, dict) or "error" in blob:
            lines.append(f"## {asset}")
            lines.append("(skipped: error or insufficient data)")
            lines.append("")
            continue

        lines.append(f"## {asset}")
        lines.append("")
        n_bars = blob.get("n_bars", 0)
        ic_by_h = blob.get("ic_by_horizon", {})
        deciles_by_h = blob.get("decile_forward_returns", {})
        json_summary[asset] = {"poor_deciles_by_horizon": {}, "strong_deciles_by_horizon": {}, "ic_by_horizon": ic_by_h}

        for h_key, h_data in deciles_by_h.items():
            if not isinstance(h_data, dict) or "deciles" not in h_data:
                continue
            horizon = h_data.get("horizon", h_key)
            deciles = h_data["deciles"]
            poor = []
            strong = []
            for d in deciles:
                q = d.get("decile")
                mean_ret = d.get("mean_forward_return", 0)
                se = d.get("std_error", 0)
                n = d.get("n_bars", 0)
                if mean_ret is None:
                    continue
                if mean_ret < -flat_thresh:
                    poor.append((q, mean_ret, se, n))
                elif mean_ret > flat_thresh:
                    strong.append((q, mean_ret, se, n))

            json_summary[asset]["poor_deciles_by_horizon"][str(horizon)] = [q for q, _, _, _ in poor]
            json_summary[asset]["strong_deciles_by_horizon"][str(horizon)] = [q for q, _, _, _ in strong]

            lines.append(f"### Horizon {horizon} bars")
            lines.append("")
            ic_val = ic_by_h.get(f"ic_{horizon}", ic_by_h.get(h_key))
            if ic_val is not None:
                lines.append(f"- **IC:** {ic_val:.4f}")
            lines.append("")
            if poor:
                lines.append("**Poor or flat deciles (consider avoiding entry in this score range):**")
                for q, mean_ret, se, n in poor:
                    lines.append(f"- Decile {q}: mean fwd return = {mean_ret:.4f} (se={se:.4f}, n={n})")
                lines.append("")
            if strong:
                lines.append("**Strong deciles (favorable Entry_Score range):**")
                for q, mean_ret, se, n in strong:
                    lines.append(f"- Decile {q}: mean fwd return = {mean_ret:.4f} (se={se:.4f}, n={n})")
                lines.append("")
            lines.append("")

        lines.append("")

    md_path = out_dir / "decile_validation_report.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report written to {md_path}")

    json_path = out_dir / "decile_validation_summary.json"
    with open(json_path, "w") as f:
        json.dump(json_summary, f, indent=2)
    print(f"Summary JSON written to {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
