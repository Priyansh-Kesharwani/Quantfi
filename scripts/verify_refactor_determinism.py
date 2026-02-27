#!/usr/bin/env python3
"""
Refactor path — determinism verification.

Runs a fixed refactor pipeline (load snapshot, normalizer + one composite path
with fixed seed), writes outputs to validation/artifacts/{runid}/, then runs
again and asserts SHA256 of key outputs (e.g. metrics.json or score series) are identical.

Usage:
    python scripts/verify_refactor_determinism.py --runid refactor_test

Exit 0 → deterministic; exit 1 → non-deterministic or error.
"""

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


def _file_hash(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def run_refactor_pipeline(seed: int, runid: str, artifacts_dir: Path) -> None:
    """Run refactor pipeline once and write artifacts to artifacts_dir."""
    np.random.seed(seed)

    from indicators.normalization_refactor import canonical_normalize

    snapshot_path = PROJECT_ROOT / "data_snapshots" / "refactor_verification.parquet"
    if snapshot_path.exists():
        df = pd.read_parquet(snapshot_path)
        if "symbol" in df.columns:
            sym = df["symbol"].iloc[0]
            df = df.loc[df["symbol"] == sym]
        if "close" in df.columns:
            raw = df["close"].values
        else:
            raw = df["Close"].values if "Close" in df.columns else df.iloc[:, 0].values
    else:
        raw = np.random.RandomState(seed).randn(500).cumsum() + 100

    s_t, meta = canonical_normalize(
        raw, k=1.0, eps=1e-9, mode="approx", higher_is_favorable=True, min_obs=50
    )
    scores = pd.Series(s_t, name="score")

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "runid": runid,
        "seed": seed,
        "n_obs": int(meta.get("n_obs", 0)),
        "mode": meta.get("mode", "approx"),
    }
    with open(artifacts_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    scores.to_csv(artifacts_dir / "scores.csv", index=True)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runid", default="refactor_test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    artifacts_base = PROJECT_ROOT / "validation" / "artifacts"
    artifacts_base.mkdir(parents=True, exist_ok=True)
    run_dir = artifacts_base / args.runid

    # Run 1
    run_refactor_pipeline(args.seed, args.runid, run_dir)
    hash1_metrics = _file_hash(run_dir / "metrics.json")
    hash1_scores = _file_hash(run_dir / "scores.csv")

    # Run 2
    run_refactor_pipeline(args.seed, args.runid, run_dir)
    hash2_metrics = _file_hash(run_dir / "metrics.json")
    hash2_scores = _file_hash(run_dir / "scores.csv")

    if hash1_metrics != hash2_metrics or hash1_scores != hash2_scores:
        print("Determinism check FAILED: hashes differ between runs", file=sys.stderr)
        return 1
    print("Determinism check passed: identical hashes across two runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
