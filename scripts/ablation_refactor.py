#!/usr/bin/env python3
"""
Refactor path — Leave-one-component-out ablation: Δ Sortino / IC.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


def main() -> int:
    from indicators.composite_refactor import compute_composite_score_refactor

    n = 100
    idx = pd.RangeIndex(n)
    np.random.seed(42)
    base = {
        "T_t": pd.Series(np.clip(np.random.rand(n) * 0.5 + 0.25, 0, 1), index=idx),
        "U_t": pd.Series(np.clip(np.random.rand(n) * 0.5 + 0.25, 0, 1), index=idx),
        "H_t": pd.Series(np.clip(np.random.rand(n) * 0.5 + 0.25, 0, 1), index=idx),
        "LDC_t": pd.Series(np.clip(np.random.rand(n) * 0.5 + 0.25, 0, 1), index=idx),
        "O_t": pd.Series(np.clip(np.random.rand(n) * 0.5 + 0.25, 0, 1), index=idx),
        "C_t": pd.Series(0.9 * np.ones(n), index=idx),
        "L_t": pd.Series(0.9 * np.ones(n), index=idx),
        "R_t": pd.Series(0.5 * np.ones(n), index=idx),
        "TBL_flag": pd.Series(np.random.rand(n), index=idx),
        "OFI_rev": pd.Series(np.random.rand(n), index=idx),
        "lambda_decay": pd.Series(np.random.rand(n), index=idx),
    }
    entry_full, _, _ = compute_composite_score_refactor(base)
    keys_abl = ["T_t", "U_t", "H_t", "LDC_t", "O_t"]
    for k in keys_abl:
        comp = {kk: vv.copy() for kk, vv in base.items()}
        comp[k] = pd.Series(0.5 * np.ones(n), index=idx)
        entry_abl, _, _ = compute_composite_score_refactor(comp)
        delta = (entry_full - entry_abl).abs().mean()
        print(f"Leave-{k}: mean |Δ score| = {delta:.4f}")
    print("Ablation refactor done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
