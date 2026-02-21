"""Minimal ROC AUC helper — avoids heavy sklearn dependency in tests."""

import numpy as np


def manual_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC AUC using the trapezoidal rule.

    Parameters
    ----------
    labels : array-like of {0, 1}
    scores : array-like of float

    Returns
    -------
    float  AUC ∈ [0, 1].
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    # Sort descending by score
    order = np.argsort(-scores)
    labels_sorted = labels[order]

    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.5

    tp = 0
    fp = 0
    auc = 0.0
    prev_fpr = 0.0

    for lab in labels_sorted:
        if lab == 1:
            tp += 1
        else:
            fp += 1
            # Whenever we see a negative, TPR changes → trapezoid
            tpr = tp / n_pos
            fpr = fp / n_neg
            auc += (fpr - prev_fpr) * tpr
            prev_fpr = fpr

    return auc
