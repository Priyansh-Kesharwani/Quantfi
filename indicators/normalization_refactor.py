"""
Canonical normalization for the refactor path.

Single source of truth for: expanding ECDF (with exact midrank tie rule)
→ inverse-normal (Phi^{-1}) → sigmoid. All refactor composite and indicator
normalization must import from this module only; no duplicate ECDF/sigmoid
logic elsewhere for the refactor path.

Formulas (see specs/REFRACTOR_MATHEMATICS.md):
  p_t = (rank_less_t + 0.5 * rank_equal_t) / n_t
  z_t = Phi^{-1}(clip(p_t, eps, 1-eps))
  s_t = 1 / (1 + exp(-k * z_t))
  If higher raw is harmful: s_t <- 1 - s_t
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from typing import Tuple, Dict, Any

from indicators.normalization import (
    expanding_percentile,
    percentile_to_z,
    z_to_sigmoid,
    polarity_align,
)


def _expanding_midrank_ecdf(raw: np.ndarray, min_obs: int) -> np.ndarray:
    """
    Expanding ECDF with exact midrank tie rule.

    p_t = (rank_less_t + 0.5 * rank_equal_t) / n_t
    where n_t = number of valid observations in raw[0:t+1].
    """
    raw = np.asarray(raw, dtype=np.float64)
    n = len(raw)
    pct_t = np.full(n, np.nan)
    for t in range(min_obs, n):
        current = raw[t]
        if np.isnan(current):
            continue
        hist = raw[: t + 1]
        valid = hist[~np.isnan(hist)]
        if len(valid) < min_obs:
            continue
        n_t = len(valid)
        rank_less = np.sum(valid < current)
        rank_equal = np.sum(valid == current)
        pct_t[t] = (rank_less + 0.5 * rank_equal) / n_t
    return pct_t


def canonical_normalize(
    raw: np.ndarray,
    k: float = 1.0,
    eps: float = 1e-9,
    mode: str = "exact",
    higher_is_favorable: bool = True,
    min_obs: int = 1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Expand ECDF → inverse-normal → sigmoid with configurable polarity.

    Parameters
    ----------
    raw : np.ndarray
        Raw values (1d).
    k : float
        Sigmoid steepness.
    eps : float
        Clip probability to [eps, 1-eps] before Phi^{-1}.
    mode : str
        "exact" = midrank tie rule (rank_less + 0.5*rank_equal)/n_t;
        "approx" = expanding_percentile-style (approximates exact).
    higher_is_favorable : bool
        If True, higher raw → higher score; if False, s_t <- 1 - s_t.
    min_obs : int
        Minimum observations before producing non-NaN.

    Returns
    -------
    s_t : np.ndarray
        Scores in (0, 1).
    meta : dict
        Keys: k, eps, mode, higher_is_favorable, n_obs, method.
    """
    raw = np.asarray(raw, dtype=np.float64)
    n = len(raw)

    if mode == "exact":
        pct_t = _expanding_midrank_ecdf(raw, min_obs)
        pct_t = np.clip(pct_t, eps, 1.0 - eps)
        z_t = stats.norm.ppf(pct_t)
        z_t = np.where(np.isnan(pct_t), np.nan, z_t)
        safe_z = np.clip(z_t * k, -500, 500)
        s_t = 1.0 / (1.0 + np.exp(-safe_z))
        s_t = np.where(np.isnan(z_t), np.nan, s_t)
        eps_f = np.finfo(np.float64).eps
        s_t = np.clip(s_t, eps_f, 1.0 - eps_f)
    elif mode == "approx":
        pct_t, _ = expanding_percentile(raw, min_obs=min_obs)
        pct_t = np.clip(pct_t, eps, 1.0 - eps)
        z_t = percentile_to_z(pct_t)
        s_t = z_to_sigmoid(z_t, k=k)
    else:
        raise NotImplementedError(
            f"canonical_normalize mode={mode!r} not implemented; use 'exact' or 'approx'."
        )

    if not higher_is_favorable:
        s_t = polarity_align(s_t, higher_is_favorable=False)

    meta: Dict[str, Any] = {
        "k": k,
        "eps": eps,
        "mode": mode,
        "higher_is_favorable": higher_is_favorable,
        "n_obs": int(np.sum(~np.isnan(s_t))),
        "method": "canonical_normalize",
    }
    return s_t, meta
