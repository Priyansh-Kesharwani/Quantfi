"""
Phase B — Walk-Forward Cross-Validation Engine.

Implements a sliding-window walk-forward validation strategy:
  - Train window: configurable (default 3 years ≈ 756 bars)
  - Test window: configurable (default 1 year ≈ 252 bars)
  - Optional overlap between segments
  - Expanding or rolling train window

Evaluates Entry_Score / Exit_Score via signal generation → backtest → metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
import logging
import json
import sys
from pathlib import Path

from engine.validation.metrics import (
    compute_all_metrics,
    forward_returns,
    information_coefficient,
    hit_rate,
    sortino_ratio,
    max_drawdown,
    cagr,
)

logger = logging.getLogger(__name__)

@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward cross-validation."""
    train_window: int = 756
    test_window: int = 252
    overlap: bool = False
    expanding: bool = True
    forward_horizons: List[int] = field(default_factory=lambda: [5, 10, 20])
    entry_threshold: float = 70.0
    exit_threshold: float = 70.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WalkForwardConfig":
        return cls(
            train_window=d.get("train_window", 756),
            test_window=d.get("test_window", 252),
            overlap=d.get("overlap", False),
            expanding=d.get("expanding", True),
            forward_horizons=d.get("forward_windows", [5, 10, 20]),
            entry_threshold=d.get("entry_threshold", 70.0),
            exit_threshold=d.get("exit_threshold", 70.0),
        )

@dataclass
class WalkForwardFoldResult:
    """Result of a single walk-forward fold."""
    fold_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_size: int
    test_size: int
    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

@dataclass
class WalkForwardResult:
    """Aggregate walk-forward validation result."""
    symbol: str
    n_folds: int
    folds: List[WalkForwardFoldResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "n_folds": self.n_folds,
            "folds": [
                {
                    "fold_idx": f.fold_idx,
                    "train_start": f.train_start,
                    "train_end": f.train_end,
                    "test_start": f.test_start,
                    "test_end": f.test_end,
                    "train_size": f.train_size,
                    "test_size": f.test_size,
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

def _generate_folds(
    n_bars: int,
    train_window: int,
    test_window: int,
    overlap: bool = False,
    expanding: bool = True,
) -> List[Tuple[int, int, int, int]]:
    """Generate (train_start, train_end, test_start, test_end) index tuples.

    Parameters
    ----------
    n_bars : int
        Total number of bars.
    train_window : int
        Number of bars in each training window.
    test_window : int
        Number of bars in each test window.
    overlap : bool
        If True, test windows can overlap.
    expanding : bool
        If True, training window expands from t=0; else it slides.

    Returns
    -------
    list of (train_start, train_end, test_start, test_end) tuples.
    """
    folds = []
    step = test_window if not overlap else test_window // 2

    test_start = train_window
    while test_start + test_window <= n_bars:
        test_end = test_start + test_window

        if expanding:
            t_start = 0
        else:
            t_start = max(0, test_start - train_window)
        t_end = test_start

        folds.append((t_start, t_end, test_start, test_end))
        test_start += step

    return folds

def walkforward_cv(
    df: pd.DataFrame,
    score_fn: Callable[[pd.DataFrame], Tuple[pd.Series, pd.Series]],
    train_window: int = 756,
    test_window: int = 252,
    overlap: bool = False,
    expanding: bool = True,
    forward_horizons: Optional[List[int]] = None,
    entry_threshold: float = 70.0,
    exit_threshold: float = 70.0,
    symbol: str = "UNKNOWN",
) -> WalkForwardResult:
    """Run walk-forward cross-validation on a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Full history with OHLCV columns. Must have 'close' column.
    score_fn : callable
        Function(df_train: pd.DataFrame) -> (entry_scores, exit_scores)
        that computes Entry_Score and Exit_Score for the given data slice.
        It receives the *test* slice but may also receive training data
        context (the function should handle warm-up internally).
    train_window : int
        Bars in training window (default 756 ≈ 3 years).
    test_window : int
        Bars in test window (default 252 ≈ 1 year).
    overlap : bool
        Allow overlapping test windows.
    expanding : bool
        Use expanding (vs. rolling) training window.
    forward_horizons : list of int
        Forward-return horizons to evaluate.
    entry_threshold : float
        Score threshold for entry signals.
    exit_threshold : float
        Score threshold for exit signals.
    symbol : str
        Asset symbol for labelling.

    Returns
    -------
    WalkForwardResult
    """
    if forward_horizons is None:
        forward_horizons = [5, 10, 20]

    col_map = {c.lower(): c for c in df.columns}
    close_col = col_map.get("close", "close")
    prices = df[close_col]

    folds_spec = _generate_folds(
        len(df), train_window, test_window, overlap, expanding
    )

    result = WalkForwardResult(symbol=symbol, n_folds=len(folds_spec))

    all_fold_metrics: List[Dict[str, Any]] = []

    for fold_i, (tr_s, tr_e, te_s, te_e) in enumerate(folds_spec):
        logger.info(
            f"  Fold {fold_i}: train=[{tr_s}:{tr_e}] ({tr_e - tr_s} bars) "
            f"test=[{te_s}:{te_e}] ({te_e - te_s} bars)"
        )

        df_train = df.iloc[tr_s:tr_e]
        df_test = df.iloc[te_s:te_e]

        try:
            df_combined = pd.concat([df_train, df_test])
            entry_scores, exit_scores = score_fn(df_combined)

            test_idx = df_test.index
            entry_test = entry_scores.reindex(test_idx)
            exit_test = exit_scores.reindex(test_idx)
            prices_test = prices.reindex(test_idx)

            fold_metrics = compute_all_metrics(
                entry_test,
                exit_test,
                prices_test,
                forward_horizons=forward_horizons,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
            )

            fold_result = WalkForwardFoldResult(
                fold_idx=fold_i,
                train_start=str(df_train.index[0]),
                train_end=str(df_train.index[-1]),
                test_start=str(df_test.index[0]),
                test_end=str(df_test.index[-1]),
                train_size=len(df_train),
                test_size=len(df_test),
                metrics=fold_metrics,
            )
            all_fold_metrics.append(fold_metrics)

        except Exception as e:
            logger.error(f"  Fold {fold_i} failed: {e}")
            fold_result = WalkForwardFoldResult(
                fold_idx=fold_i,
                train_start=str(df_train.index[0]) if len(df_train) > 0 else "N/A",
                train_end=str(df_train.index[-1]) if len(df_train) > 0 else "N/A",
                test_start=str(df_test.index[0]) if len(df_test) > 0 else "N/A",
                test_end=str(df_test.index[-1]) if len(df_test) > 0 else "N/A",
                train_size=len(df_train),
                test_size=len(df_test),
                warnings=[str(e)],
            )

        result.folds.append(fold_result)

    result.summary = _summarise_folds(all_fold_metrics, forward_horizons)
    return result

def _summarise_folds(
    all_metrics: List[Dict[str, Any]],
    forward_horizons: List[int],
) -> Dict[str, Any]:
    """Aggregate fold-level metrics into summary statistics."""
    if not all_metrics:
        return {}

    summary: Dict[str, Any] = {}

    for h in forward_horizons:
        ic_key = f"ic_{h}d"
        hr_key = f"hit_rate_{h}d"
        ics = [m["entry_metrics"].get(ic_key, np.nan) for m in all_metrics]
        hrs = [m["entry_metrics"].get(hr_key, np.nan) for m in all_metrics]
        summary[f"mean_{ic_key}"] = float(np.nanmean(ics))
        summary[f"std_{ic_key}"] = float(np.nanstd(ics))
        summary[f"mean_{hr_key}"] = float(np.nanmean(hrs))

    sortinos = [m["entry_metrics"].get("sortino", np.nan) for m in all_metrics]
    cagrs = [m["entry_metrics"].get("cagr", np.nan) for m in all_metrics]
    dds = [m["entry_metrics"].get("max_drawdown", np.nan) for m in all_metrics]

    summary["mean_sortino"] = float(np.nanmean(sortinos))
    summary["mean_cagr"] = float(np.nanmean(cagrs))
    summary["mean_max_drawdown"] = float(np.nanmean(dds))

    n_trades = [m["signal_metrics"].get("n_trades", 0) for m in all_metrics]
    rois = [m["signal_metrics"].get("roi_per_trade", np.nan) for m in all_metrics]
    wlrs = [m["signal_metrics"].get("win_loss_ratio", np.nan) for m in all_metrics]

    summary["total_trades"] = int(sum(n_trades))
    summary["mean_roi_per_trade"] = float(np.nanmean(rois))
    summary["mean_win_loss_ratio"] = float(np.nanmean(wlrs))

    return summary
