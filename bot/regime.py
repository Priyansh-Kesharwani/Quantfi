"""
HMM-based regime detection: fit_hmm and predict_state_prob (posterior R_t).
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any, Optional

try:
    from hmmlearn import hmm
except ImportError as e:
    raise ImportError(
        "bot.regime requires hmmlearn. Install with: pip install hmmlearn"
    ) from e


def fit_hmm(
    returns_df: pd.DataFrame,
    n_states: int = 3,
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """
    Fit a Gaussian HMM on returns.

    Parameters
    ----------
    returns_df : pd.DataFrame
        At least one column of returns (e.g. close.pct_change()); uses first column if multiple.
    n_states : int
        Number of hidden states.
    random_state : int, optional
        For reproducibility.
    **kwargs
        Passed to GaussianHMM (e.g. covariance_type="full", n_iter=100).

    Returns
    -------
    Fitted GaussianHMM model (from hmmlearn).
    """
    if returns_df.shape[1] >= 1:
        returns = returns_df.iloc[:, 0].astype(float).dropna().values.reshape(-1, 1)
    else:
        raise ValueError("returns_df must have at least one column")
    if len(returns) < n_states * 2:
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


def predict_state_prob(
    df: pd.DataFrame,
    model: Any,
    returns_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Return posterior state probabilities R_t aligned to df.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with index to align; must contain returns (or first numeric column used).
    model : fitted GaussianHMM
        From fit_hmm().
    returns_column : str, optional
        Column name for returns; if None, use first column.

    Returns
    -------
    pd.DataFrame
        Columns state_0, state_1, ... (posterior probabilities); index = df.index.
    """
    if returns_column and returns_column in df.columns:
        returns = df[returns_column].astype(float)
    else:
        returns = df.iloc[:, 0].astype(float)
    valid = returns.notna()
    X = returns.values[valid].reshape(-1, 1)
    probs = model.predict_proba(X)
    n_states = probs.shape[1]
    out = pd.DataFrame(
        np.nan,
        index=df.index,
        columns=[f"state_{i}" for i in range(n_states)],
    )
    out.loc[valid, :] = probs
    return out
