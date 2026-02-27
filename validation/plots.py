"""
Phase B — Visualization Utilities.

Generates publication-quality plots for the validation report:
  - Score distribution histograms
  - Walk-forward summary tables (as image)
  - Exit signal ROC curve
  - λ(t) simulation overlays
  - Entry_Score vs forward return heatmaps
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging
import io
import base64

logger = logging.getLogger(__name__)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def _fig_to_base64(fig: plt.Figure, dpi: int = 100) -> str:
    """Convert a matplotlib figure to a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded

def _fig_to_file(fig: plt.Figure, path: str, dpi: int = 150) -> str:
    """Save a matplotlib figure to a file and return the path."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_score_distribution(
    entry_scores: pd.Series,
    exit_scores: pd.Series,
    title: str = "Score Distributions",
    bins: int = 50,
    save_path: Optional[str] = None,
) -> str:
    """Plot histograms of Entry_Score and Exit_Score.

    Returns
    -------
    str
        Base64-encoded PNG (or file path if save_path given).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(
        entry_scores.dropna(), bins=bins, color="#2196F3",
        alpha=0.7, edgecolor="white"
    )
    axes[0].set_title("Entry Score Distribution")
    axes[0].set_xlabel("Entry Score")
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(70, color="red", linestyle="--", alpha=0.6, label="Threshold=70")
    axes[0].legend()

    axes[1].hist(
        exit_scores.dropna(), bins=bins, color="#FF9800",
        alpha=0.7, edgecolor="white"
    )
    axes[1].set_title("Exit Score Distribution")
    axes[1].set_xlabel("Exit Score")
    axes[1].set_ylabel("Frequency")
    axes[1].axvline(70, color="red", linestyle="--", alpha=0.6, label="Threshold=70")
    axes[1].legend()

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save_path:
        return _fig_to_file(fig, save_path)
    return _fig_to_base64(fig)

def plot_walkforward_summary(
    wf_result_dict: Dict[str, Any],
    save_path: Optional[str] = None,
) -> str:
    """Render walk-forward fold results as a table figure.

    Parameters
    ----------
    wf_result_dict : dict
        Output from WalkForwardResult.to_dict().

    Returns
    -------
    str
        Base64 PNG or file path.
    """
    folds = wf_result_dict.get("folds", [])
    if not folds:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No folds available", ha="center", va="center")
        ax.axis("off")
        if save_path:
            return _fig_to_file(fig, save_path)
        return _fig_to_base64(fig)

    rows = []
    for f in folds:
        m = f.get("metrics", {})
        em = m.get("entry_metrics", {})
        sm = m.get("signal_metrics", {})
        rows.append([
            f["fold_idx"],
            f"{f['test_start'][:10]} → {f['test_end'][:10]}",
            f"{em.get('ic_5d', float('nan')):.3f}",
            f"{em.get('hit_rate_5d', float('nan')):.1%}",
            f"{em.get('sortino', float('nan')):.2f}",
            f"{em.get('max_drawdown', float('nan')):.1%}",
            f"{sm.get('n_trades', 0)}",
        ])

    col_labels = ["Fold", "Test Period", "IC(5d)", "Hit%(5d)", "Sortino", "MaxDD", "#Trades"]

    fig, ax = plt.subplots(figsize=(14, max(3, 0.5 * len(rows) + 1)))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#1565C0")
        table[0, j].set_text_props(color="white", fontweight="bold")

    fig.suptitle(
        f"Walk-Forward CV — {wf_result_dict.get('symbol', '')}",
        fontsize=12, fontweight="bold"
    )
    fig.tight_layout()

    if save_path:
        return _fig_to_file(fig, save_path)
    return _fig_to_base64(fig)

def plot_hawkes_overlay(
    grid: np.ndarray,
    true_intensity: np.ndarray,
    estimated_intensity: Optional[np.ndarray] = None,
    events: Optional[np.ndarray] = None,
    regime_name: str = "",
    rmse: Optional[float] = None,
    save_path: Optional[str] = None,
) -> str:
    """Plot true λ(t) vs estimated λ(t) with event markers.

    Parameters
    ----------
    grid : np.ndarray
        Time grid.
    true_intensity : np.ndarray
        Ground-truth λ(t).
    estimated_intensity : np.ndarray or None
        Estimated λ(t) overlay.
    events : np.ndarray or None
        Event timestamps for scatter markers.
    regime_name : str
        Regime label for title.
    rmse : float or None
        RMSE value to display.

    Returns
    -------
    str
        Base64 PNG or file path.
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(grid, true_intensity, color="#1565C0", linewidth=1.5,
            label="True λ(t)", alpha=0.9)

    if estimated_intensity is not None:
        min_len = min(len(grid), len(estimated_intensity))
        ax.plot(grid[:min_len], estimated_intensity[:min_len],
                color="#FF5722", linewidth=1.2, linestyle="--",
                label="Estimated λ(t)", alpha=0.8)

    if events is not None and len(events) > 0:
        ax.scatter(events, np.zeros_like(events), marker="|",
                   color="gray", alpha=0.4, s=30, label="Events")

    title = f"Hawkes Intensity — {regime_name}"
    if rmse is not None:
        title += f" (RMSE={rmse:.4f})"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("λ(t)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        return _fig_to_file(fig, save_path)
    return _fig_to_base64(fig)

def plot_score_return_heatmap(
    scores: pd.Series,
    forward_returns: pd.Series,
    n_score_bins: int = 10,
    n_ret_bins: int = 10,
    title: str = "Entry Score vs Forward Return",
    save_path: Optional[str] = None,
) -> str:
    """2D histogram heatmap of score vs forward return.

    Returns
    -------
    str
        Base64 PNG or file path.
    """
    mask = scores.notna() & forward_returns.notna()
    s = scores[mask].values
    r = forward_returns[mask].values

    if len(s) < 10:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Insufficient data for heatmap",
                ha="center", va="center")
        ax.axis("off")
        if save_path:
            return _fig_to_file(fig, save_path)
        return _fig_to_base64(fig)

    fig, ax = plt.subplots(figsize=(8, 6))

    h, xedges, yedges = np.histogram2d(s, r, bins=[n_score_bins, n_ret_bins])
    im = ax.imshow(
        h.T, origin="lower", aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap="YlOrRd",
    )
    plt.colorbar(im, ax=ax, label="Count")
    ax.set_xlabel("Entry Score")
    ax.set_ylabel("Forward Return")
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()

    if save_path:
        return _fig_to_file(fig, save_path)
    return _fig_to_base64(fig)

def plot_exit_roc(
    exit_scores: pd.Series,
    true_regime_flips: pd.Series,
    title: str = "Exit Score ROC Curve",
    save_path: Optional[str] = None,
) -> str:
    """Plot ROC curve for exit signal vs true regime flips.

    Parameters
    ----------
    exit_scores : pd.Series
        Exit_Score values (continuous).
    true_regime_flips : pd.Series
        Binary labels (1 = regime flip occurred).

    Returns
    -------
    str
        Base64 PNG or file path.
    """
    mask = exit_scores.notna() & true_regime_flips.notna()
    scores = exit_scores[mask].values
    labels = true_regime_flips[mask].values.astype(int)

    if len(scores) < 10 or len(np.unique(labels)) < 2:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.text(0.5, 0.5, "Insufficient data for ROC",
                ha="center", va="center")
        ax.axis("off")
        if save_path:
            return _fig_to_file(fig, save_path)
        return _fig_to_base64(fig)

    thresholds = np.sort(np.unique(scores))[::-1]
    tpr_list = []
    fpr_list = []

    for thresh in thresholds:
        preds = (scores >= thresh).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    fpr_arr = np.array([0.0] + fpr_list + [1.0])
    tpr_arr = np.array([0.0] + tpr_list + [1.0])

    order = np.argsort(fpr_arr)
    fpr_arr = fpr_arr[order]
    tpr_arr = tpr_arr[order]

    auc = float(np.trapz(tpr_arr, fpr_arr))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr_arr, tpr_arr, color="#1565C0", linewidth=2,
            label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()

    if save_path:
        return _fig_to_file(fig, save_path)
    return _fig_to_base64(fig)

def plot_equity_curve(
    equity: pd.Series,
    benchmark: Optional[pd.Series] = None,
    title: str = "Equity Curve",
    save_path: Optional[str] = None,
) -> str:
    """Plot equity curve with optional benchmark overlay.

    Returns
    -------
    str
        Base64 PNG or file path.
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(equity.index, equity.values, color="#1565C0",
            linewidth=1.5, label="Strategy")

    if benchmark is not None:
        ax.plot(benchmark.index, benchmark.values, color="gray",
                linewidth=1, linestyle="--", alpha=0.7, label="Benchmark")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        return _fig_to_file(fig, save_path)
    return _fig_to_base64(fig)
