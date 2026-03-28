"""
Bot scoring API: composite formula → CompositeScore_t, EntryScore_t, ExitScore_t, breakdown.

Delegates to indicators.composite_refactor; exposes normalized components,
g_pers(H_t), Opp_t, Gate_t, RawFavor_t, and per-component breakdown.
"""

from __future__ import annotations

import pandas as pd
from typing import Dict, Any, Optional, Tuple

from engine.indicators.composite_refactor import (
    compute_composite_score_refactor,
    load_refactor_config,
)


def compute_composite_scores(
    components: Dict[str, pd.Series],
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    """
    Compute composite, entry, exit scores and per-component breakdown.

    Components should be normalized [0,1] Series: T_t, U_t, O_t (OFI), H_t,
    LDC_t, C_t, L_t, R_t (regime). For exit: TBL_flag, OFI_rev, lambda_decay
    (optional; defaulted to 0.5 if missing).

    Returns
    -------
    composite_score : pd.Series
        CompositeScore_t in [0, 100].
    entry_score : pd.Series
        EntryScore_t (same as composite for entry signal).
    exit_score : pd.Series
        ExitScore_t in [0, 100] from exit formula.
    breakdown : pd.DataFrame
        Per-component: Opp_t, Gate_t, RawFavor_t, and score columns.
    """
    if config is None:
        try:
            config = load_refactor_config()
        except Exception:
            config = {
                "composite": {"trim_frac": 0.1, "r_thresh": 0.2, "S_scale": 1.0, "k_pers": 6.0},
                "exit": {"gamma1": 1.0, "gamma2": 1.0, "gamma3": 1.0},
            }
    entry_series, exit_series, breakdown_df = compute_composite_score_refactor(
        components, config=config
    )
    composite_score = entry_series.rename("CompositeScore_t")
    entry_score = entry_series.rename("EntryScore_t")
    exit_score = exit_series.rename("ExitScore_t")
    breakdown_df = breakdown_df.rename(columns={"RawFavor": "RawFavor_t"})
    return composite_score, entry_score, exit_score, breakdown_df
