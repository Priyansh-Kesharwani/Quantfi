"""
Combinatorial Purged Cross-Validation (CPCV).

Sequential partition into N folds; combinatorial splits (k_test folds as test);
purging (event-window overlap with test) and embargo (fixed bars after test).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class CPCVSplit:
    train_idx: np.ndarray
    test_idx: np.ndarray
    split_id: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CPCVConfig:
    n_folds: int = 5
    k_test: int = 2
    horizon_bars: int = 20
    embargo_bars: int = 24
    seed: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CPCVConfig":
        return cls(
            n_folds=int(d.get("n_folds", 5)),
            k_test=int(d.get("k_test", 2)),
            horizon_bars=int(d.get("horizon_bars", 20)),
            embargo_bars=int(d.get("embargo_bars", 24)),
            seed=d.get("seed"),
        )


def _fold_boundaries(n_samples: int, n_folds: int) -> np.ndarray:
    fold_sizes = np.full(n_folds, n_samples // n_folds)
    fold_sizes[: n_samples % n_folds] += 1
    return np.cumsum(np.r_[0, fold_sizes])


def _test_ranges_for_combo(
    fold_starts: np.ndarray,
    n_samples: int,
    test_fold_indices: tuple,
) -> List[tuple]:
    ranges = []
    for fi in test_fold_indices:
        start = int(fold_starts[fi])
        end = int(fold_starts[fi + 1])
        ranges.append((start, end))
    return ranges


def _overlaps(t_start: int, t_end: int, test_ranges: List[tuple]) -> bool:
    for a, b in test_ranges:
        if not (t_end <= a or t_start >= b):
            return True
    return False


def generate_cpcv_splits(
    config: CPCVConfig,
    n_samples: int,
) -> List[CPCVSplit]:
    """
    Generate CPCV splits: N folds, k_test folds as test per split, purge + embargo.

    Parameters
    ----------
    config : CPCVConfig
        n_folds, k_test, horizon_bars, embargo_bars, seed (optional).
    n_samples : int
        Total number of time-ordered samples.

    Returns
    -------
    List[CPCVSplit]
        One entry per combination of k_test folds as test; train purged and embargoed.
    """
    n_folds = config.n_folds
    k_test = config.k_test
    horizon_bars = config.horizon_bars
    embargo_bars = config.embargo_bars

    if n_folds < k_test:
        return []
    fold_starts = _fold_boundaries(n_samples, n_folds)
    indices = np.arange(n_samples)
    out: List[CPCVSplit] = []
    split_id = 0
    for test_fold_tuple in combinations(range(n_folds), k_test):
        test_ranges = _test_ranges_for_combo(fold_starts, n_samples, test_fold_tuple)
        test_idx = np.concatenate([
            indices[int(fold_starts[fi]) : int(fold_starts[fi + 1])]
            for fi in test_fold_tuple
        ])
        test_set = set(test_idx.tolist())
        train_mask = np.ones(n_samples, dtype=bool)
        for i in range(n_samples):
            if i in test_set:
                train_mask[i] = False
                continue
            t_start, t_end = i, min(i + horizon_bars, n_samples)
            if _overlaps(t_start, t_end, test_ranges):
                train_mask[i] = False
        for a, b in test_ranges:
            embargo_start = b
            embargo_end = min(b + embargo_bars, n_samples)
            train_mask[embargo_start:embargo_end] = False
        train_idx = indices[train_mask]
        if len(train_idx) == 0:
            continue
        out.append(
            CPCVSplit(
                train_idx=train_idx,
                test_idx=test_idx,
                split_id=split_id,
                metadata={
                    "test_folds": list(test_fold_tuple),
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                },
            )
        )
        split_id += 1
    return out


def write_splits_metadata(
    path: Path,
    splits: List[CPCVSplit],
    date_index: Optional[np.ndarray] = None,
) -> None:
    """
    Write split metadata to JSON for downstream validation scripts.
    date_index: optional array of dates (e.g. pd.DatetimeIndex.values).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for s in splits:
        rec = {
            "split_id": s.split_id,
            "n_train": len(s.train_idx),
            "n_test": len(s.test_idx),
            "train_start": int(s.train_idx[0]),
            "train_end": int(s.train_idx[-1]),
            "test_start": int(s.test_idx[0]),
            "test_end": int(s.test_idx[-1]),
            "metadata": s.metadata,
        }
        if date_index is not None and len(date_index) > max(s.test_idx[-1], s.train_idx[-1]):
            rec["test_start_date"] = str(date_index[int(s.test_idx[0])])
            rec["test_end_date"] = str(date_index[int(s.test_idx[-1])])
        records.append(rec)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
