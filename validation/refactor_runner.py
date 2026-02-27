"""
Refactor path — single runner that writes validation/artifacts/{runid}/.

Loads snapshot or data, runs refactor pipeline (normalizer + composite),
writes scores.parquet, breakdown.parquet, metrics.json.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Optional
import numpy as np
import pandas as pd


def run_refactor_pipeline(
    runid: str = "refactor_run",
    seed: int = 42,
    snapshot_path: Optional[Path] = None,
) -> Path:
    """Run refactor pipeline and write artifacts to validation/artifacts/{runid}/."""
    from indicators.normalization_refactor import canonical_normalize
    from indicators.composite_refactor import compute_composite_score_refactor

    np.random.seed(seed)
    artifacts_dir = PROJECT_ROOT / "validation" / "artifacts" / runid
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if snapshot_path is None:
        snapshot_path = PROJECT_ROOT / "data_snapshots" / "refactor_verification.parquet"
    if not snapshot_path.exists():
        # Synthetic data
        raw = np.random.RandomState(seed).randn(500).cumsum() + 100
        n = len(raw)
        idx = pd.RangeIndex(n)
    else:
        df = pd.read_parquet(snapshot_path)
        if "symbol" in df.columns:
            sym = df["symbol"].iloc[0]
            df = df.loc[df["symbol"] == sym]
        raw = df["close"].values if "close" in df.columns else df["Close"].values
        n = len(raw)
        idx = df.index

    s_t, norm_meta = canonical_normalize(raw, k=1.0, mode="approx", min_obs=50)
    n_obs = int(norm_meta.get("n_obs", 0))

    components = {
        "T_t": pd.Series(0.5 * np.ones(n), index=idx),
        "U_t": pd.Series(s_t, index=idx).fillna(0.5),
        "H_t": pd.Series(0.5 * np.ones(n), index=idx),
        "LDC_t": pd.Series(0.5 * np.ones(n), index=idx),
        "O_t": pd.Series(s_t, index=idx).fillna(0.5),
        "C_t": pd.Series(0.9 * np.ones(n), index=idx),
        "L_t": pd.Series(0.9 * np.ones(n), index=idx),
        "R_t": pd.Series(0.5 * np.ones(n), index=idx),
        "TBL_flag": pd.Series(0.5 * np.ones(n), index=idx),
        "OFI_rev": pd.Series(0.5 * np.ones(n), index=idx),
        "lambda_decay": pd.Series(0.5 * np.ones(n), index=idx),
    }
    entry, exit_s, breakdown = compute_composite_score_refactor(components)

    metrics = {"runid": runid, "seed": seed, "n_obs": n_obs, "n_bars": n}
    with open(artifacts_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame({"Entry_Score": entry, "Exit_Score": exit_s}).to_parquet(artifacts_dir / "scores.parquet", index=True)
    breakdown.to_parquet(artifacts_dir / "breakdown.parquet", index=True)
    return artifacts_dir


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runid", default="refactor_run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snapshot", type=str, default=None)
    args = parser.parse_args()
    path = args.snapshot
    run_refactor_pipeline(args.runid, args.seed, Path(path) if path else None)
    print(f"Artifacts written to validation/artifacts/{args.runid}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
