#!/usr/bin/env python3
"""
Inspect tuning outputs: gt_scores, tuning_trials, winning_config.
Prints per-config per-split summary, penalty counts, and suggested param ranges.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

UNDERPERFORM_PENALTY = -1e6
PENALTY_THRESHOLD = -1e5


def main():
    ap = argparse.ArgumentParser(description="Inspect GT scores and tuning results")
    ap.add_argument("--gt-scores", default="validation/outputs/gt_scores.json", help="Path to gt_scores.json")
    ap.add_argument("--trials", default="validation/outputs/tuning_trials.json", help="Path to tuning_trials.json")
    ap.add_argument("--winning", default="validation/outputs/winning_config.json", help="Path to winning_config.json")
    ap.add_argument("--config", default=None, help="Optional YAML config path; used to check for cpcv_splits_metadata.json in output dir")
    ap.add_argument("--out", default=None, help="Write report to file instead of stdout")
    args = ap.parse_args()

    out_lines = []

    def emit(s=""):
        out_lines.append(s)

    gt_path = Path(args.gt_scores)
    if not gt_path.is_file():
        emit(f"Missing: {gt_path}")
        _write_out(args.out, out_lines)
        return 1

    with open(gt_path) as f:
        gt_matrix = json.load(f)

    emit("=== GT scores (per config) ===")
    by_config = defaultdict(list)
    for r in gt_matrix:
        by_config[r["config_key"]].append(r["gt_score"])

    for cfg_key, scores in sorted(by_config.items()):
        arr = [float(s) for s in scores]
        mean_gt = sum(arr) / len(arr) if arr else 0
        variance = sum((x - mean_gt) ** 2 for x in arr) / len(arr) if len(arr) > 1 else 0
        std_gt = variance ** 0.5
        n_penalty = sum(1 for s in arr if s <= PENALTY_THRESHOLD)
        emit(f"  {cfg_key}")
        emit(f"    mean_gt={mean_gt:.2f} std={std_gt:.2f} min={min(arr):.2f} max={max(arr):.2f} n_splits={len(arr)} penalty_count={n_penalty}")

    penalty_total = sum(1 for r in gt_matrix if float(r.get("gt_score", 0)) <= PENALTY_THRESHOLD)
    emit("")
    emit(f"=== Penalty summary ===")
    emit(f"  Underperform penalty applied in {penalty_total} of {len(gt_matrix)} split-config pairs (gt_score <= {PENALTY_THRESHOLD})")

    trials_path = Path(args.trials)
    if trials_path.is_file():
        with open(trials_path) as f:
            trials = json.load(f)
        emit("")
        emit("=== Trials ===")
        best_score = None
        best_trial_id = None
        for t in trials:
            s = t.get("score")
            if best_score is None or (s is not None and s > best_score):
                best_score = s
                best_trial_id = t.get("trial_id")
        for t in trials:
            mark = " [BEST]" if t.get("trial_id") == best_trial_id else ""
            emit(f"  trial_id={t.get('trial_id')} config={t.get('config')} score={t.get('score')} compute_time_s={t.get('compute_time_s')}{mark}")
    else:
        emit("")
        emit(f"  (no {trials_path})")

    winning_path = Path(args.winning)
    if winning_path.is_file():
        with open(winning_path) as f:
            winning = json.load(f)
        emit("")
        emit("=== Winning config ===")
        emit(f"  config: {winning.get('config')}")
        emit(f"  dsr: {winning.get('dsr')}")
        emit(f"  mean_gt_score: {winning.get('mean_gt_score')}")
        emit(f"  dsr_passed: {winning.get('dsr_passed')}")
        emit(f"  n_trials: {winning.get('n_trials')}")
    else:
        emit("")
        emit(f"  (no {winning_path})")

    out_dir = gt_path.parent
    if args.config:
        config_path = Path(args.config)
        if config_path.is_file():
            try:
                import yaml
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)
                emit("")
                emit("=== Config (data/cpcv) ===")
                emit(f"  data: {cfg.get('data', {})}")
                emit(f"  cpcv: {cfg.get('cpcv', {})}")
            except Exception as e:
                emit(f"  (could not load config: {e})")
        splits_meta = out_dir / "cpcv_splits_metadata.json"
        if splits_meta.is_file():
            with open(splits_meta) as f:
                meta = json.load(f)
            emit("")
            emit("=== CPCV splits metadata (test windows) ===")
            for m in meta[:5]:
                emit(f"  {m}")
            if len(meta) > 5:
                emit(f"  ... and {len(meta) - 5} more")

    emit("")
    emit("=== Suggested next param ranges ===")
    if not by_config:
        emit("  No configs in gt_scores.")
    else:
        best_mean = max((sum(s) / len(s), cfg) for cfg, s in by_config.items() if s)
        if best_mean[0] <= PENALTY_THRESHOLD:
            emit("  All configs hit underperform penalty. Suggest widening:")
            emit("    entry_score_threshold: [50, 55, 60, 65, 70, 75, 80]")
            emit("    max_positions: [3, 5, 10, 15, 20]")
        else:
            best_cfg = best_mean[1]
            emit(f"  Best mean GT at config: {best_cfg}")
            try:
                cfg = json.loads(best_cfg)
                entry = cfg.get("entry_score_threshold")
                max_pos = cfg.get("max_positions")
                if entry is not None:
                    low = max(50, entry - 10)
                    high = min(90, entry + 10)
                    emit(f"  Suggest concentrating: entry_score_threshold: [{low}, {entry-5}, {entry}, {entry+5}, {high}]")
                if max_pos is not None:
                    emit(f"  Suggest around: max_positions: [{max(1, max_pos-5)}, {max_pos}, {min(30, max_pos+5)}]")
            except Exception:
                pass

    _write_out(args.out, out_lines)
    return 0


def _write_out(out_path, lines):
    text = "\n".join(lines)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    sys.exit(main())
