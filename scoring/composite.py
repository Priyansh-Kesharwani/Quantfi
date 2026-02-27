"""
Shared composite scoring logic used by both the backend ScoringEngine
and the backtester's vectorized scorer.

Single source of truth for scoring rules, thresholds, and category functions.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


# ── Canonical rule parameters ────────────────────────────────────
# These match the defaults in BackendConfig.  Both backend/scoring.py
# and backtester/portfolio_simulator.py should import these constants
# so the two paths never drift.

SCORING_RULES = {
    "base_score": 50.0,

    # Technical momentum
    "sma_below_bonus": 15.0,
    "sma_above_penalty": 5.0,
    "rsi_oversold": 30.0,
    "rsi_low": 40.0,
    "rsi_overbought": 70.0,
    "rsi_high": 60.0,
    "rsi_oversold_bonus": 20.0,
    "rsi_low_bonus": 10.0,
    "rsi_overbought_penalty": 15.0,
    "rsi_high_penalty": 5.0,
    "macd_bull_bonus": 10.0,
    "macd_bear_penalty": 10.0,
    "bb_below_bonus": 15.0,
    "bb_near_multiplier": 1.02,
    "bb_near_bonus": 8.0,
    "adx_low": 20.0,
    "adx_high": 40.0,
    "adx_low_bonus": 5.0,
    "adx_high_penalty": 5.0,

    # Volatility opportunity
    "atr_high_percentile": 80.0,
    "atr_mid_percentile": 60.0,
    "atr_low_percentile": 30.0,
    "atr_high_bonus": 20.0,
    "atr_mid_bonus": 10.0,
    "atr_low_penalty": 10.0,
    "drawdown_severe": -20.0,
    "drawdown_medium": -10.0,
    "drawdown_light": -5.0,
    "drawdown_flat": -1.0,
    "drawdown_severe_bonus": 30.0,
    "drawdown_medium_bonus": 20.0,
    "drawdown_light_bonus": 10.0,
    "drawdown_flat_penalty": 10.0,

    # Statistical deviation
    "extreme_low_z": -2.0,
    "strong_low_z": -1.5,
    "moderate_low_z": -1.0,
    "light_low_z": -0.5,
    "high_z": 1.5,
    "moderate_high_z": 1.0,
    "extreme_low_bonus": 50.0,
    "strong_low_bonus": 35.0,
    "moderate_low_bonus": 20.0,
    "light_low_bonus": 10.0,
    "high_penalty": 30.0,
    "moderate_high_penalty": 15.0,

    # Macro FX
    "fx_historical_avg": 83.0,
    "fx_high_dev": 5.0,
    "fx_mid_dev": 2.0,
    "fx_low_dev": -2.0,
    "fx_very_low_dev": -5.0,
    "fx_high_penalty": 20.0,
    "fx_mid_penalty": 10.0,
    "fx_low_bonus": 10.0,
    "fx_very_low_bonus": 20.0,
}

DEFAULT_WEIGHTS: Dict[str, float] = {
    "technical_momentum": 0.4,
    "volatility_opportunity": 0.2,
    "statistical_deviation": 0.2,
    "macro_fx": 0.2,
}


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def technical_momentum_score(
    current_price: float,
    sma_200: Optional[float],
    rsi: Optional[float],
    macd_line: Optional[float],
    macd_hist: Optional[float],
    bb_lower: Optional[float],
    bb_upper: Optional[float],
    adx: Optional[float],
    rules: Optional[Dict[str, float]] = None,
) -> float:
    """Score a single bar's technical momentum (0-100)."""
    r = rules or SCORING_RULES
    score = r["base_score"]

    if sma_200 is not None and not np.isnan(sma_200):
        if current_price < sma_200:
            score += r["sma_below_bonus"]
        else:
            score -= r["sma_above_penalty"]

    if rsi is not None and not np.isnan(rsi):
        if rsi < r["rsi_oversold"]:
            score += r["rsi_oversold_bonus"]
        elif rsi < r["rsi_low"]:
            score += r["rsi_low_bonus"]
        elif rsi > r["rsi_overbought"]:
            score -= r["rsi_overbought_penalty"]
        elif rsi > r["rsi_high"]:
            score -= r["rsi_high_penalty"]

    if macd_line is not None and macd_hist is not None and not (np.isnan(macd_line) or np.isnan(macd_hist)):
        if macd_line < 0 and macd_hist > 0:
            score += r["macd_bull_bonus"]
        elif macd_line > 0 and macd_hist < 0:
            score -= r["macd_bear_penalty"]

    if bb_lower is not None and bb_upper is not None and not (np.isnan(bb_lower) or np.isnan(bb_upper)):
        if current_price < bb_lower:
            score += r["bb_below_bonus"]
        elif current_price < bb_lower * r["bb_near_multiplier"]:
            score += r["bb_near_bonus"]

    if adx is not None and not np.isnan(adx):
        if adx < r["adx_low"]:
            score += r["adx_low_bonus"]
        elif adx > r["adx_high"]:
            score -= r["adx_high_penalty"]

    return _clamp(score)


def volatility_opportunity_score(
    atr_percentile: Optional[float],
    drawdown_pct: Optional[float],
    rules: Optional[Dict[str, float]] = None,
) -> float:
    """Score a single bar's volatility opportunity (0-100)."""
    r = rules or SCORING_RULES
    score = r["base_score"]

    if atr_percentile is not None and not np.isnan(atr_percentile):
        if atr_percentile > r["atr_high_percentile"]:
            score += r["atr_high_bonus"]
        elif atr_percentile > r["atr_mid_percentile"]:
            score += r["atr_mid_bonus"]
        elif atr_percentile < r["atr_low_percentile"]:
            score -= r["atr_low_penalty"]

    if drawdown_pct is not None and not np.isnan(drawdown_pct):
        if drawdown_pct < r["drawdown_severe"]:
            score += r["drawdown_severe_bonus"]
        elif drawdown_pct < r["drawdown_medium"]:
            score += r["drawdown_medium_bonus"]
        elif drawdown_pct < r["drawdown_light"]:
            score += r["drawdown_light_bonus"]
        elif drawdown_pct > r["drawdown_flat"]:
            score -= r["drawdown_flat_penalty"]

    return _clamp(score)


def statistical_deviation_score(
    avg_z_score: Optional[float],
    rules: Optional[Dict[str, float]] = None,
) -> float:
    """Score a single bar's statistical deviation (0-100)."""
    r = rules or SCORING_RULES
    score = r["base_score"]

    if avg_z_score is not None and not np.isnan(avg_z_score):
        if avg_z_score < r["extreme_low_z"]:
            score += r["extreme_low_bonus"]
        elif avg_z_score < r["strong_low_z"]:
            score += r["strong_low_bonus"]
        elif avg_z_score < r["moderate_low_z"]:
            score += r["moderate_low_bonus"]
        elif avg_z_score < r["light_low_z"]:
            score += r["light_low_bonus"]
        elif avg_z_score > r["high_z"]:
            score -= r["high_penalty"]
        elif avg_z_score > r["moderate_high_z"]:
            score -= r["moderate_high_penalty"]

    return _clamp(score)


def macro_fx_score(
    usd_inr_rate: float,
    rules: Optional[Dict[str, float]] = None,
) -> float:
    """Score macro FX conditions (0-100)."""
    r = rules or SCORING_RULES
    score = r["base_score"]
    historical_avg = r["fx_historical_avg"]
    dev_pct = ((usd_inr_rate - historical_avg) / historical_avg) * 100

    if dev_pct > r["fx_high_dev"]:
        score -= r["fx_high_penalty"]
    elif dev_pct > r["fx_mid_dev"]:
        score -= r["fx_mid_penalty"]
    elif dev_pct < r["fx_very_low_dev"]:
        score += r["fx_very_low_bonus"]
    elif dev_pct < r["fx_low_dev"]:
        score += r["fx_low_bonus"]

    return _clamp(score)


def compute_composite_score_single(
    current_price: float,
    sma_200: Optional[float],
    rsi: Optional[float],
    macd_line: Optional[float],
    macd_hist: Optional[float],
    bb_lower: Optional[float],
    bb_upper: Optional[float],
    adx: Optional[float],
    atr_percentile: Optional[float],
    drawdown_pct: Optional[float],
    avg_z_score: Optional[float],
    usd_inr_rate: float = 83.5,
    weights: Optional[Dict[str, float]] = None,
    rules: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, float]]:
    """Compute composite score for a single bar.

    Returns (composite_score, breakdown_dict).
    """
    w = weights or DEFAULT_WEIGHTS
    tech = technical_momentum_score(current_price, sma_200, rsi, macd_line, macd_hist, bb_lower, bb_upper, adx, rules)
    vol = volatility_opportunity_score(atr_percentile, drawdown_pct, rules)
    stat = statistical_deviation_score(avg_z_score, rules)
    fx = macro_fx_score(usd_inr_rate, rules)

    composite = (
        tech * w.get("technical_momentum", 0.4)
        + vol * w.get("volatility_opportunity", 0.2)
        + stat * w.get("statistical_deviation", 0.2)
        + fx * w.get("macro_fx", 0.2)
    )

    breakdown = {
        "technical_momentum": tech,
        "volatility_opportunity": vol,
        "statistical_deviation": stat,
        "macro_fx": fx,
    }
    return composite, breakdown
