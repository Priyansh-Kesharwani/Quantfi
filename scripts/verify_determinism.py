#!/usr/bin/env python3
"""
Phase A — Determinism Verification Script.

Runs the full Phase A indicator pipeline twice with identical inputs
and verifies that outputs are bit-for-bit identical by comparing
SHA-256 hashes of the resulting DataFrames.

Usage:
    python scripts/verify_determinism.py

Exit code 0  → deterministic
Exit code 1  → non-deterministic (or error)
"""

import hashlib
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from indicators.normalization import expanding_ecdf_sigmoid
from indicators.ofi import compute_ofi, compute_ofi_reversal
from indicators.hawkes import estimate_hawkes, hawkes_lambda_decay
from indicators.ldc import LDC, build_templates_from_labels
from indicators.composite import compose_scores, PhaseAConfig
from tests.fixtures import fbm_series, ou_series, hawkes_events

def _dataframe_hash(df: pd.DataFrame) -> str:
    """Compute SHA-256 of a DataFrame's bytes."""
    buf = df.to_csv(index=True).encode("utf-8")
    return hashlib.sha256(buf).hexdigest()

def _series_hash(s: pd.Series) -> str:
    return hashlib.sha256(s.to_csv(index=True).encode("utf-8")).hexdigest()

def run_pipeline(seed: int = 42) -> dict:
    """Execute the full Phase A pipeline deterministically.

    Returns a dict of name → SHA-256 hash for every computed artefact.
    Also returns a second dict of the raw Series/arrays for snapshot use.
    """
    df = fbm_series(n=400, H=0.7, seed=seed)

    norm_close = expanding_ecdf_sigmoid(df["close"], k=1.0, polarity=1, min_obs=50)

    ofi = compute_ofi(df, window=20, normalize=True, min_obs=50)
    ofi_rev = compute_ofi_reversal(df, window=20, min_obs=50)

    h_events = hawkes_events(mu=0.5, alpha=0.3, beta=1.0, T=100.0, seed=seed)
    timestamps = np.arange(0, 100, 1.0)
    intensity, hawkes_meta = estimate_hawkes({"trades": h_events}, timestamps)
    lam_decay = hawkes_lambda_decay(
        {"trades": h_events}, timestamps, min_obs=20,
    )

    np.random.seed(seed)
    n_feat = 5
    bull_feat = np.random.randn(50, n_feat) + 1.0
    bear_feat = np.random.randn(50, n_feat) - 1.0
    features = np.vstack([bull_feat, bear_feat])
    labels = np.array([1] * 50 + [0] * 50)
    templates = build_templates_from_labels(features, labels)
    ldc = LDC(kappa=1.0)
    ldc.fit(templates)
    ldc_scores = ldc.score_batch(features)

    n = len(df)
    idx = df.index
    np.random.seed(seed)
    components = {
        "T_t": pd.Series(np.random.uniform(0.3, 0.7, n), index=idx),
        "U_t": pd.Series(np.random.uniform(0.3, 0.7, n), index=idx),
        "H_t": pd.Series(np.random.uniform(0.4, 0.6, n), index=idx),
        "R_t": pd.Series(np.random.uniform(0.4, 0.8, n), index=idx),
        "C_t": pd.Series(np.random.uniform(0.5, 0.9, n), index=idx),
        "OFI_t": ofi,
        "P_move_t": pd.Series(np.random.uniform(0.3, 0.7, n), index=idx),
        "S_t": pd.Series(np.zeros(n), index=idx),
        "TBL_flag": pd.Series(np.random.uniform(0.2, 0.8, n), index=idx),
        "OFI_rev": ofi_rev,
        "lambda_decay": pd.Series(np.random.uniform(0.2, 0.8, n), index=idx),
    }
    entry, exit_, breakdown = compose_scores(components)

    hashes = {
        "norm_close": _series_hash(norm_close),
        "ofi": _series_hash(ofi),
        "ofi_rev": _series_hash(ofi_rev),
        "intensity": _series_hash(intensity),
        "lam_decay": _series_hash(lam_decay),
        "ldc_scores": hashlib.sha256(ldc_scores.tobytes()).hexdigest(),
        "entry": _series_hash(entry),
        "exit": _series_hash(exit_),
        "breakdown": _dataframe_hash(breakdown),
    }

    raw = {
        "norm_close": norm_close,
        "ofi": ofi,
        "ofi_rev": ofi_rev,
        "entry": entry,
        "exit": exit_,
    }

    return hashes, raw

def main():
    print("=" * 60)
    print("Phase A — Determinism Verification")
    print("=" * 60)

    print("\n🔄 Run 1 ...")
    h1, raw1 = run_pipeline(seed=42)

    print("🔄 Run 2 ...")
    h2, _raw2 = run_pipeline(seed=42)

    all_ok = True
    for key in h1:
        match = h1[key] == h2[key]
        status = "✅" if match else "❌"
        print(f"  {status} {key:20s}  {h1[key][:16]}...")
        if not match:
            all_ok = False

    print()
    if all_ok:
        print("✅ All outputs are deterministic (identical hashes).")

        snap_dir = PROJECT_ROOT / "data_snapshots"
        snap_dir.mkdir(exist_ok=True)
        snap_path = snap_dir / "verification_snapshot.parquet"

        df = fbm_series(n=400, H=0.7, seed=42)
        snap = df.copy()
        for col_name, series in raw1.items():
            snap[col_name] = series.values
        snap.to_parquet(str(snap_path))
        print(f"📦 Snapshot saved to {snap_path}")

        sys.exit(0)
    else:
        print("❌ FAILED — non-deterministic output detected!")
        sys.exit(1)

if __name__ == "__main__":
    main()
