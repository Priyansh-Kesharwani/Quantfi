"""
Phase B — Purged K-Fold Cross-Validation (Non-Leaky).

Implements time-series-aware K-fold splitting with embargo zones to
prevent information leakage between train and test folds.

Steps:
  1. Split time-ordered bars into K folds (preserving temporal order)
  2. Remove overlap around each test fold (embargo = configurable bars)
  3. Run composite score engine → Entry/Exit → scorer function
  4. Record fold-level metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable, Tuple, Iterator
from dataclasses import dataclass, field
import logging
import json

from validation.metrics import (
    compute_all_metrics,
    forward_returns,
    information_coefficient,
    hit_rate,
    evaluate_signals,
)

logger = logging.getLogger(__name__)

def _purged_kfold_splits(
    n_samples: int,
    n_splits: int,
    embargo: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate purged K-fold train/test index arrays.

    Parameters
    ----------
    n_samples : int
        Total number of samples.
    n_splits : int
        Number of folds.
    embargo : int
        Number of bars to remove around each test fold boundary.

    Returns
    -------
    list of (train_idx, test_idx) tuples.
    """
    indices = np.arange(n_samples)
    fold_sizes = np.full(n_splits, n_samples // n_splits)
    fold_sizes[:n_samples % n_splits] += 1
    fold_starts = np.cumsum(np.r_[0, fold_sizes])

    splits = []
    for i in range(n_splits):
        test_start = fold_starts[i]
        test_end = fold_starts[i + 1]
        test_idx = indices[test_start:test_end]

        embargo_end = min(test_end + embargo, n_samples)
        embargo_start = max(test_start - embargo, 0)

        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[embargo_start:embargo_end] = False
        train_idx = indices[train_mask]

        if len(train_idx) == 0:
            logger.warning(f"Fold {i}: empty training set after purge+embargo")
            continue

        splits.append((train_idx, test_idx))

    return splits

@dataclass
class PurgedKFoldConfig:
    """Configuration for purged K-fold cross-validation."""
    n_splits: int = 5
    embargo_bars: int = 20
    forward_horizons: List[int] = field(default_factory=lambda: [5, 10, 20])
    entry_threshold: float = 70.0
    exit_threshold: float = 70.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PurgedKFoldConfig":
        return cls(
            n_splits=d.get("n_splits", 5),
            embargo_bars=d.get("embargo_bars", 20),
            forward_horizons=d.get("forward_windows", [5, 10, 20]),
            entry_threshold=d.get("entry_threshold", 70.0),
            exit_threshold=d.get("exit_threshold", 70.0),
        )

@dataclass
class KFoldFoldResult:
    """Result of a single K-fold split."""
    fold_idx: int
    train_size: int
    test_size: int
    test_start: str
    test_end: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

@dataclass
class KFoldResult:
    """Aggregate purged K-fold result."""
    symbol: str
    n_splits: int
    embargo: int
    folds: List[KFoldFoldResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "n_splits": self.n_splits,
            "embargo": self.embargo,
            "folds": [
                {
                    "fold_idx": f.fold_idx,
                    "train_size": f.train_size,
                    "test_size": f.test_size,
                    "test_start": f.test_start,
                    "test_end": f.test_end,
                    "metrics": f.metrics,
                    "warnings": f.warnings,
                }
                for f in self.folds
            ],
            "summary": self.summary,
        }

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

def purged_kfold(
    df: pd.DataFrame,
    score_fn: Callable[[pd.DataFrame], Tuple[pd.Series, pd.Series]],
    n_splits: int = 5,
    embargo: int = 20,
    forward_horizons: Optional[List[int]] = None,
    entry_threshold: float = 70.0,
    exit_threshold: float = 70.0,
    symbol: str = "UNKNOWN",
) -> KFoldResult:
    """Run purged K-fold cross-validation.

    Parameters
    ----------
    df : pd.DataFrame
        Full history with OHLCV columns.
    score_fn : callable
        Function(df_slice: pd.DataFrame) -> (entry_scores, exit_scores).
        Called with the *full* DataFrame for context, but metrics are
        computed only on the test fold.
    n_splits : int
        Number of folds.
    embargo : int
        Number of bars to purge around test fold boundaries.
    forward_horizons : list of int
        Forward-return horizons for IC and hit rate.
    entry_threshold : float
        Score threshold for entry signals.
    exit_threshold : float
        Score threshold for exit signals.
    symbol : str
        Asset symbol.

    Returns
    -------
    KFoldResult
    """
    if forward_horizons is None:
        forward_horizons = [5, 10, 20]

    col_map = {c.lower(): c for c in df.columns}
    close_col = col_map.get("close", "close")
    prices = df[close_col]

    splits = _purged_kfold_splits(len(df), n_splits, embargo)

    result = KFoldResult(
        symbol=symbol,
        n_splits=n_splits,
        embargo=embargo,
    )

    all_fold_metrics: List[Dict[str, Any]] = []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        logger.info(
            f"  Fold {fold_i}: train={len(train_idx)} bars, "
            f"test={len(test_idx)} bars, embargo={embargo}"
        )

        df_test = df.iloc[test_idx]

        try:
            entry_scores, exit_scores = score_fn(df)

            test_index = df_test.index
            entry_test = entry_scores.reindex(test_index)
            exit_test = exit_scores.reindex(test_index)
            prices_test = prices.reindex(test_index)

            fold_metrics = compute_all_metrics(
                entry_test,
                exit_test,
                prices_test,
                forward_horizons=forward_horizons,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
            )

            fold_result = KFoldFoldResult(
                fold_idx=fold_i,
                train_size=len(train_idx),
                test_size=len(test_idx),
                test_start=str(df_test.index[0]),
                test_end=str(df_test.index[-1]),
                metrics=fold_metrics,
            )
            all_fold_metrics.append(fold_metrics)

        except Exception as e:
            logger.error(f"  Fold {fold_i} failed: {e}")
            fold_result = KFoldFoldResult(
                fold_idx=fold_i,
                train_size=len(train_idx),
                test_size=len(test_idx),
                test_start=str(df_test.index[0]) if len(df_test) > 0 else "N/A",
                test_end=str(df_test.index[-1]) if len(df_test) > 0 else "N/A",
                warnings=[str(e)],
            )

        result.folds.append(fold_result)

    result.summary = _summarise_kfold(all_fold_metrics, forward_horizons)
    return result

def _summarise_kfold(
    all_metrics: List[Dict[str, Any]],
    forward_horizons: List[int],
) -> Dict[str, Any]:
    """Aggregate K-fold metrics into summary statistics."""
    if not all_metrics:
        return {}

    summary: Dict[str, Any] = {}

    for h in forward_horizons:
        ics = [m["entry_metrics"].get(f"ic_{h}d", np.nan) for m in all_metrics]
        hrs = [m["entry_metrics"].get(f"hit_rate_{h}d", np.nan) for m in all_metrics]
        summary[f"mean_ic_{h}d"] = float(np.nanmean(ics))
        summary[f"std_ic_{h}d"] = float(np.nanstd(ics))
        summary[f"mean_hit_rate_{h}d"] = float(np.nanmean(hrs))

    n_trades = [m["signal_metrics"].get("n_trades", 0) for m in all_metrics]
    rois = [m["signal_metrics"].get("roi_per_trade", np.nan) for m in all_metrics]
    pfs = [m["signal_metrics"].get("profit_factor", np.nan) for m in all_metrics]

    summary["total_trades"] = int(sum(n_trades))
    summary["mean_roi_per_trade"] = float(np.nanmean(rois))
    summary["mean_profit_factor"] = float(np.nanmean(pfs))

    if len(all_metrics) >= 2:
        all_ics = []
        for h in forward_horizons:
            all_ics.extend([
                m["entry_metrics"].get(f"ic_{h}d", np.nan)
                for m in all_metrics
            ])
        valid_ics = [x for x in all_ics if not np.isnan(x)]
        summary["ic_cross_fold_std"] = float(np.std(valid_ics)) if valid_ics else np.nan
    else:
        summary["ic_cross_fold_std"] = np.nan

    return summary
