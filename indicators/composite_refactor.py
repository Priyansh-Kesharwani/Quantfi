"""
Refactor-path composite: Entry/Exit from phase_refactor.yml only.

Formulas (see specs/REFRACTOR_MATHEMATICS.md):
  Step A: g_pers(H_t) = sigmoid(k_pers * (H_t - 0.5))
  Step B: All components already normalized [0,1], higher = favorable
  Step C: Opp_t = trimmed_mean(T_t, U_t * g_pers(H_t), LDC_t, O_t)
  Step D: Gate_t = C_t * L_t * 1(R_t >= r_thresh)
  Step E: RawFavor_t = Opp_t * Gate_t; CompositeScore_t = 100 * clip(0.5 + (RawFavor_t - 0.5) * S_scale, 0, 1)
  Exit: ExitRaw_t = γ1*TBL_t + γ2*OFI_rev_t + γ3*lambda_decay_t; normalize → ExitScore_t in [0,100]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from indicators.normalization_refactor import canonical_normalize


def load_refactor_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load config/phase_refactor.yml."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "config" / "phase_refactor.yml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def g_pers_refactor(H_t: np.ndarray, k_pers: float) -> np.ndarray:
    """g_pers(H_t) = sigmoid(k_pers * (H_t - 0.5)), in (0, 1)."""
    H = np.asarray(H_t, dtype=np.float64)
    z = k_pers * (H - 0.5)
    z = np.clip(z, -500, 500)
    return np.where(np.isnan(H), np.nan, 1.0 / (1.0 + np.exp(-z)))


def _trimmed_mean_row(values: np.ndarray, trim_frac: float) -> float:
    """Trimmed mean of a 1d array; trim_frac in [0, 0.5)."""
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return np.nan
    if trim_frac <= 0:
        return float(np.mean(valid))
    from scipy.stats import trim_mean
    return float(trim_mean(valid, trim_frac))


def compute_opportunity_refactor(
    components: Dict[str, pd.Series],
    trim_frac: float,
    k_pers: float,
    opp_keys: Optional[list] = None,
) -> pd.Series:
    """
    Opp_t = trimmed_mean(T_t, U_t * g_pers(H_t), LDC_t, O_t).
    components: dict of Series [0,1]; must contain H_t for g_pers.
    """
    if opp_keys is None:
        opp_keys = ["T_t", "U_t", "H_t", "LDC_t", "O_t"]
    idx = next(iter(components.values())).index
    n = len(idx)
    # Default 0.5 for missing
    get = lambda k: components.get(k, pd.Series(np.full(n, 0.5), index=idx))
    H_t = get("H_t")
    U_t = get("U_t")
    g_H = pd.Series(g_pers_refactor(H_t.values, k_pers), index=idx)
    U_weighted = U_t * g_H

    # Build matrix: T_t, U_weighted, LDC_t, O_t (and optionally others)
    use_keys = [k for k in opp_keys if k not in ("H_t",)]
    rows = []
    for k in use_keys:
        if k == "U_t":
            rows.append(U_weighted.values)
        else:
            rows.append(get(k).values)
    if not rows:
        return pd.Series(np.full(n, 0.5), index=idx, name="Opp_t")
    mat = np.column_stack(rows)
    opp = np.array([_trimmed_mean_row(mat[i], trim_frac) for i in range(n)])
    return pd.Series(opp, index=idx, name="Opp_t")


def compute_gate_refactor(
    components: Dict[str, pd.Series],
    r_thresh: float,
) -> pd.Series:
    """Gate_t = C_t * L_t * 1(R_t >= r_thresh)."""
    idx = next(iter(components.values())).index
    n = len(idx)
    get = lambda k: components.get(k, pd.Series(np.full(n, 0.5), index=idx))
    C_t = get("C_t")
    L_t = get("L_t")
    R_t = get("R_t")
    R_pass = (R_t >= r_thresh).astype(float)
    gate = C_t * L_t * R_pass
    return gate.rename("Gate_t")


def compute_composite_score_refactor(
    components: Dict[str, pd.Series],
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Entry CompositeScore and Exit score from refactor config.
    components: dict of normalized [0,1] Series (T_t, U_t, H_t, LDC_t, O_t, C_t, L_t, R_t for entry;
                TBL_flag, OFI_rev, lambda_decay for exit).
    """
    if config is None:
        config = load_refactor_config()
    comp_cfg = config.get("composite", {})
    exit_cfg = config.get("exit", {})
    trim_frac = comp_cfg.get("trim_frac", 0.1)
    r_thresh = comp_cfg.get("r_thresh", 0.2)
    S_scale = comp_cfg.get("S_scale", 1.0)
    k_pers = comp_cfg.get("k_pers", 6.0)
    gamma1 = exit_cfg.get("gamma1", 1.0)
    gamma2 = exit_cfg.get("gamma2", 1.0)
    gamma3 = exit_cfg.get("gamma3", 1.0)

    idx = next(iter(components.values())).index
    n = len(idx)

    Opp_t = compute_opportunity_refactor(components, trim_frac, k_pers)
    Gate_t = compute_gate_refactor(components, r_thresh)
    RawFavor = Opp_t * Gate_t
    transformed = 0.5 + (RawFavor - 0.5) * S_scale
    Entry_Score = 100.0 * pd.Series(np.clip(transformed.values, 0.0, 1.0), index=idx, name="CompositeScore")

    # Exit: ExitRaw = γ1*TBL + γ2*OFI_rev + γ3*lambda_decay, then normalize to [0,100]
    get = lambda k: components.get(k, pd.Series(np.full(n, 0.5), index=idx))
    TBL_flag = get("TBL_flag")
    OFI_rev = get("OFI_rev")
    lambda_decay = get("lambda_decay")
    ExitRaw = gamma1 * TBL_flag + gamma2 * OFI_rev + gamma3 * lambda_decay
    exit_arr = ExitRaw.values
    exit_norm, _ = canonical_normalize(exit_arr, k=1.0, mode="approx", min_obs=max(1, n // 10))
    Exit_Score = 100.0 * pd.Series(exit_norm, index=idx, name="Exit_Score")

    breakdown = pd.DataFrame({
        "Opp_t": Opp_t.values,
        "Gate_t": Gate_t.values,
        "RawFavor": RawFavor.values,
        "Entry_Score": Entry_Score.values,
        "Exit_Score": Exit_Score.values,
    }, index=idx)

    return Entry_Score, Exit_Score, breakdown


def composite_refactor_from_config(
    components: Dict[str, pd.Series],
    config_path: Optional[str] = None,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Load phase_refactor.yml and compute Entry/Exit scores."""
    config = load_refactor_config(config_path)
    return compute_composite_score_refactor(components, config)
