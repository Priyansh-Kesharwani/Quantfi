from __future__ import annotations

from typing import Union

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

LEAKAGE_MSG_BOUNDS = "leakage_guard: index must be 0 <= index < len(series) (past-only)."
LEAKAGE_MSG_SLICE = "leakage_guard: slice (start, end) must satisfy end <= current_index + 1 (no future data)."
LEAKAGE_MSG_GLOBAL = "leakage_guard: statistics must be computed from data with end <= current index (no global full-series stats)."


def assert_past_only(
    series: Union[np.ndarray, "pd.Series"],
    index: int,
    *,
    allow_slice: bool = True,
) -> None:
    if hasattr(series, "__len__"):
        n = len(series)
    else:
        raise TypeError("series must have length")
    if not isinstance(index, (int, np.integer)):
        raise TypeError("index must be integer")
    if index < 0 or index >= n:
        raise ValueError(LEAKAGE_MSG_BOUNDS)
    if not allow_slice:
        return
    if HAS_PANDAS and isinstance(series, pd.Series):
        if series.index is not None and hasattr(series.index, "__len__") and len(series.index) != n:
            raise ValueError("series index length mismatch")


def assert_slice_past_only(current_index: int, start: int, end: int) -> None:
    if end > current_index + 1:
        raise ValueError(LEAKAGE_MSG_SLICE)
    if start < 0 or start > end:
        raise ValueError("leakage_guard: invalid slice start/end.")
