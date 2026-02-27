"""
Adaptive, regime-aware short-term trading bot.

Modules: ingestion, features, scoring, regime, bayes_online, execution.
"""

from bot.ingestion import build_bars
from bot.features import compute_ofi, estimate_hawkes, compute_atr, LDC
from bot.scoring import compute_composite_scores
from bot.regime import fit_hmm, predict_state_prob, regime_probability_rolling
from bot.bayes_online import BayesOnline, BayesHierarchical
from bot.execution import TBLManager, estimate_ou_params, var_future

__all__ = [
    "build_bars",
    "compute_ofi",
    "estimate_hawkes",
    "compute_atr",
    "LDC",
    "compute_composite_scores",
    "fit_hmm",
    "predict_state_prob",
    "regime_probability_rolling",
    "BayesOnline",
    "BayesHierarchical",
    "TBLManager",
    "estimate_ou_params",
    "var_future",
]
