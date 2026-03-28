from __future__ import annotations

import warnings as _warnings
import pandas as pd
import numpy as np
from typing import Any, Optional

try:
    from hmmlearn import hmm
except ImportError as e:
    raise ImportError(
        "bot.regime requires hmmlearn. Install with: pip install hmmlearn"
    ) from e

HMM_MIN_OBS = 2
DEFAULT_REFIT_EVERY = 63


def fit_hmm(
    returns_df: pd.DataFrame,
    n_states: int = 3,
    random_state: Optional[int] = None,
    *,
    max_idx: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """Fit a GaussianHMM on *past* data only.

    Parameters
    ----------
    max_idx : int, optional
        If provided, only rows ``[:max_idx]`` are used for fitting so that
        no future data leaks into the model.  Callers running inside a
        walk-forward or CPCV loop should pass the current bar index here.
    """
    if returns_df.shape[1] < 1:
        raise ValueError("returns_df must have at least one column")

    df_slice = returns_df if max_idx is None else returns_df.iloc[:max_idx]

    if max_idx is None:
        _warnings.warn(
            "fit_hmm called without max_idx — fitting on full series may leak future data. "
            "Prefer regime_probability_rolling() for walk-forward safety.",
            stacklevel=2,
        )

    returns = df_slice.iloc[:, 0].astype(float).dropna().values.reshape(-1, 1)
    if len(returns) < n_states * HMM_MIN_OBS:
        raise ValueError("Not enough observations to fit HMM")
    model = hmm.GaussianHMM(
        n_components=n_states,
        random_state=random_state,
        covariance_type=kwargs.get("covariance_type", "diag"),
        n_iter=kwargs.get("n_iter", 100),
        tol=kwargs.get("tol", 1e-3),
    )
    model.fit(returns)
    return model


def _fit_hmm_on_past(
    returns_1d: np.ndarray,
    end_idx: int,
    n_states: int,
    window: Optional[int],
    random_state: Optional[int],
    **kwargs: Any,
) -> Any:
    """Fit HMM using only data up to *end_idx* (exclusive), with optional rolling window."""
    if window is not None and window > 0:
        start = max(0, end_idx - window)
        past = returns_1d[start:end_idx]
    else:
        past = returns_1d[:end_idx]
    past = past[~np.isnan(past)].reshape(-1, 1)
    if len(past) < n_states * HMM_MIN_OBS:
        return None
    model = hmm.GaussianHMM(
        n_components=n_states,
        random_state=random_state,
        covariance_type=kwargs.get("covariance_type", "diag"),
        n_iter=kwargs.get("n_iter", 100),
        tol=kwargs.get("tol", 1e-3),
    )
    model.fit(past)
    return model


def regime_probability_rolling(
    returns_df: pd.DataFrame,
    n_states: int = 3,
    window: Optional[int] = 252,
    refit_every: Optional[int] = None,
    random_state: Optional[int] = None,
    returns_column: Optional[str] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    if returns_column and returns_column in returns_df.columns:
        returns = returns_df[returns_column].astype(float)
    else:
        returns = returns_df.iloc[:, 0].astype(float)
    arr = returns.values
    n = len(arr)
    refit = refit_every if refit_every is not None else DEFAULT_REFIT_EVERY
    out = pd.DataFrame(
        np.nan,
        index=returns_df.index,
        columns=[f"state_{i}" for i in range(n_states)],
    )
    last_model = None
    last_refit_i = -1
    for i in range(n):
        if np.isnan(arr[i]):
            continue
        need_refit = last_model is None or (refit > 0 and (i - last_refit_i) >= refit)
        if need_refit:
            last_model = _fit_hmm_on_past(
                arr, i + 1, n_states, window, random_state, **kwargs
            )
            last_refit_i = i
        if last_model is None:
            continue
        try:
            probs = last_model.predict_proba(arr[i : i + 1].reshape(-1, 1))
            out.iloc[i, :] = probs[0]
        except Exception:
            pass
    return out


def predict_state_prob(
    df: pd.DataFrame,
    model: Any,
    returns_column: Optional[str] = None,
    *,
    max_idx: Optional[int] = None,
) -> pd.DataFrame:
    """Predict state probabilities.

    Parameters
    ----------
    max_idx : int, optional
        If provided, only rows ``[:max_idx]`` are scored so that future
        data is never passed through the model.
    """
    if returns_column and returns_column in df.columns:
        returns = df[returns_column].astype(float)
    else:
        returns = df.iloc[:, 0].astype(float)

    if max_idx is not None:
        returns = returns.iloc[:max_idx]

    valid = returns.notna()
    X = returns.values[valid].reshape(-1, 1)
    probs = model.predict_proba(X)
    n_states = probs.shape[1]
    out = pd.DataFrame(
        np.nan,
        index=df.index if max_idx is None else df.index[:max_idx],
        columns=[f"state_{i}" for i in range(n_states)],
    )
    out.loc[valid, :] = probs
    if max_idx is not None:
        full_out = pd.DataFrame(
            np.nan,
            index=df.index,
            columns=[f"state_{i}" for i in range(n_states)],
        )
        full_out.loc[out.index] = out
        return full_out
    return out
