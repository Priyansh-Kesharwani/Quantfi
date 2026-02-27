"""
Portfolio Simulation Engine — Mean-Reversion Strategy Backtester

Architecture (from research spec + friction-point fixes):
  1. DataPreparer      — Fetch, align, forward-fill multi-asset data; warm-up guard
  2. VectorizedScorer  — 100× faster numpy scoring (same rules as ScoringEngine)
  3. ExitFramework     — ATR-trailing stop + score-threshold + time-decay (OR logic)
  4. CostModel         — Retail-calibrated per-asset-class (IN_EQ, US_EQ, COMMODITY, CRYPTO)
  5. PortfolioSimulator— Day-by-day state machine with next-day-open execution
  6. BenchmarkRunner   — Buy-and-hold, uniform periodic investment
  7. PortfolioAnalytics— Sharpe, Sortino, Calmar, CAGR, drawdown, regime analysis
"""

import csv
import warnings

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date
from copy import deepcopy
import logging
import time

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# 1. COST MODEL
# ════════════════════════════════════════════════════════════════

COST_PRESETS = {
    "IN_EQ":          {"bps_round_trip": 40,  "fixed_per_trade_inr": 20},
    "US_EQ_FROM_IN":  {"bps_round_trip": 140, "fixed_per_trade_inr": 0},
    "US_EQ_DIRECT":   {"bps_round_trip": 20,  "fixed_per_trade_inr": 0},
    "COMMODITY":      {"bps_round_trip": 30,  "fixed_per_trade_inr": 0},
    "CRYPTO":         {"bps_round_trip": 80,  "fixed_per_trade_inr": 0},
    "INDEX":          {"bps_round_trip": 40,  "fixed_per_trade_inr": 0},
    "NONE":           {"bps_round_trip": 0,   "fixed_per_trade_inr": 0},
}


def resolve_cost_class(asset_type: str, currency: str, cost_class_override: Optional[str] = None) -> str:
    """Map (asset_type, currency) → cost class key.

    If *cost_class_override* is a valid preset key it takes precedence.
    """
    if cost_class_override and cost_class_override in COST_PRESETS:
        return cost_class_override
    at = (asset_type or "equity").lower()
    cur = (currency or "USD").upper()
    if at == "crypto":
        return "CRYPTO"
    if at == "commodity":
        return "COMMODITY"
    if at in ("index",):
        return "INDEX"
    if cur == "INR":
        return "IN_EQ"
    return "US_EQ_FROM_IN"


SLIPPAGE_MODEL_FIXED = "fixed"
SLIPPAGE_MODEL_SIZE_DEPENDENT = "size_dependent"
DEFAULT_REFERENCE_NOTIONAL = 100_000.0


def execution_price(
    raw_price: float,
    side: str,
    slippage_bps: float = 5.0,
    order_notional: Optional[float] = None,
    reference_notional: float = DEFAULT_REFERENCE_NOTIONAL,
    slippage_model: str = SLIPPAGE_MODEL_FIXED,
    slippage_impact_k: float = 10.0,
    slippage_impact_gamma: float = 0.5,
) -> float:
    if slippage_bps <= 0 and (slippage_model == SLIPPAGE_MODEL_FIXED or order_notional is None):
        return raw_price
    if slippage_model == SLIPPAGE_MODEL_SIZE_DEPENDENT and order_notional is not None and reference_notional > 0:
        ratio = min(1.0, order_notional / reference_notional)
        impact_bps = slippage_impact_k * (ratio ** slippage_impact_gamma)
        effective_bps = slippage_bps + impact_bps
    else:
        effective_bps = slippage_bps
    mult = 1 + (effective_bps / 10_000) * (1 if side == "buy" else -1)
    return raw_price * mult


def validate_slippage_dry_run(
    raw_price: float = 100.0,
    reference_notional: float = 100_000.0,
    base_bps: float = 5.0,
    k: float = 10.0,
    gamma: float = 0.5,
) -> Dict[str, Any]:
    small_notional = reference_notional * 0.01
    large_notional = reference_notional * 0.5
    p_fix_small = execution_price(raw_price, "buy", base_bps, slippage_model=SLIPPAGE_MODEL_FIXED)
    p_fix_large = execution_price(raw_price, "buy", base_bps, slippage_model=SLIPPAGE_MODEL_FIXED)
    p_size_small = execution_price(
        raw_price, "buy", base_bps,
        order_notional=small_notional,
        reference_notional=reference_notional,
        slippage_model=SLIPPAGE_MODEL_SIZE_DEPENDENT,
        slippage_impact_k=k,
        slippage_impact_gamma=gamma,
    )
    p_size_large = execution_price(
        raw_price, "buy", base_bps,
        order_notional=large_notional,
        reference_notional=reference_notional,
        slippage_model=SLIPPAGE_MODEL_SIZE_DEPENDENT,
        slippage_impact_k=k,
        slippage_impact_gamma=gamma,
    )
    slip_fix = p_fix_small - raw_price
    slip_size_small = p_size_small - raw_price
    slip_size_large = p_size_large - raw_price
    ok = slip_size_large >= slip_size_small >= slip_fix and slip_fix > 0
    return {
        "fixed_slippage": slip_fix,
        "size_dependent_small_order_slippage": slip_size_small,
        "size_dependent_large_order_slippage": slip_size_large,
        "passed": ok,
    }


def trade_cost(notional: float, cost_class: str, side: str = "entry", cost_free: bool = False) -> float:
    """Compute one-side transaction cost (half the round-trip bps + half fixed).
    If cost_free is True, returns 0 regardless of cost_class."""
    if cost_free:
        return 0.0
    cfg = COST_PRESETS.get(cost_class, COST_PRESETS["US_EQ_FROM_IN"])
    pct_cost = notional * (cfg["bps_round_trip"] / 20_000)  # half per side
    fixed = cfg["fixed_per_trade_inr"] / 2.0
    return pct_cost + fixed


# ════════════════════════════════════════════════════════════════
# 2. VECTORIZED SCORER  (same rules as ScoringEngine, 100× faster)
#    Rule constants imported from scoring.composite to prevent drift.
# ════════════════════════════════════════════════════════════════

try:
    from scoring.composite import SCORING_RULES as _SR, DEFAULT_WEIGHTS as _DW
except ImportError:
    _SR = None
    _DW = None


def vectorized_scores(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    open_: np.ndarray,
    *,
    sma_short: int = 50,
    sma_long: int = 200,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bb_period: int = 20,
    bb_std: float = 2.0,
    atr_period: int = 14,
    atr_pctl_lookback: int = 252,
    z_periods: Tuple[int, ...] = (20, 50, 100),
    adx_period: int = 14,
    usd_inr_rate: float = 83.5,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute composite scores + raw ATR for every bar using vectorised numpy ops.

    Returns
    -------
    scores : ndarray (n,)   — composite score 0-100
    atr    : ndarray (n,)   — raw ATR(14) per bar (needed for exit stops)
    """
    if weights is None:
        weights = _DW if _DW is not None else {
            "technical_momentum": 0.4,
            "volatility_opportunity": 0.2,
            "statistical_deviation": 0.2,
            "macro_fx": 0.2,
        }

    n = len(close)
    s_close = pd.Series(close)
    s_high = pd.Series(high)
    s_low = pd.Series(low)

    # ── Pre-compute indicator series ───────────────────────────
    sma50 = s_close.rolling(sma_short).mean().values
    sma200 = s_close.rolling(sma_long).mean().values
    # RSI
    delta = s_close.diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_period).mean().values
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(loss != 0, gain / loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # MACD
    ema_f = s_close.ewm(span=macd_fast, adjust=False).mean().values
    ema_s = s_close.ewm(span=macd_slow, adjust=False).mean().values
    macd_line = ema_f - ema_s
    macd_sig = pd.Series(macd_line).ewm(span=macd_signal, adjust=False).mean().values
    macd_hist = macd_line - macd_sig
    # Bollinger
    bb_mid = s_close.rolling(bb_period).mean().values
    bb_std_arr = s_close.rolling(bb_period).std().values
    bb_upper = bb_mid + bb_std * bb_std_arr
    bb_lower = bb_mid - bb_std * bb_std_arr
    # ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_arr = pd.Series(true_range).rolling(atr_period).mean().values
    # ATR percentile
    atr_pctl = pd.Series(atr_arr).rolling(min(atr_pctl_lookback, max(30, n // 4))).apply(
        lambda x: pd.Series(x).rank().iloc[-1] / len(x) * 100, raw=False
    ).values
    # Z-scores
    zs = []
    for zp in z_periods:
        rm = s_close.rolling(zp).mean().values
        rs_ = s_close.rolling(zp).std().values
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(rs_ != 0, (close - rm) / rs_, 0)
        zs.append(z)
    if zs and len(zs[0]) > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            avg_z = np.nanmean(zs, axis=0)
        avg_z = np.where(np.isnan(avg_z), 0.0, avg_z)
    else:
        avg_z = np.zeros_like(close)
    # Drawdown
    running_max = np.maximum.accumulate(close)
    drawdown = np.where(running_max > 0, (close - running_max) / running_max * 100, 0)
    # ADX (simplified vectorised)
    high_diff = s_high.diff().values
    low_diff = (-s_low.diff()).values
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    atr_adx = pd.Series(true_range).rolling(adx_period).mean().values
    plus_di = np.where(atr_adx != 0, 100 * pd.Series(plus_dm).rolling(adx_period).mean().values / atr_adx, 0)
    minus_di = np.where(atr_adx != 0, 100 * pd.Series(minus_dm).rolling(adx_period).mean().values / atr_adx, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dx = np.where(
            (plus_di + minus_di) != 0,
            100 * np.abs(plus_di - minus_di) / (plus_di + minus_di),
            0,
        )
    adx = pd.Series(dx).rolling(adx_period).mean().values

    # ── Vectorised scoring rules ──────────────────────────────
    _nan = np.isnan

    # TECHNICAL MOMENTUM
    tech = np.full(n, 50.0)
    valid_sma200 = ~_nan(sma200)
    tech = np.where(valid_sma200 & (close < sma200), tech + 15.0, tech)
    tech = np.where(valid_sma200 & (close >= sma200), tech - 5.0, tech)
    valid_rsi = ~_nan(rsi)
    tech = np.where(valid_rsi & (rsi < 30), tech + 20.0, tech)
    tech = np.where(valid_rsi & (rsi >= 30) & (rsi < 40), tech + 10.0, tech)
    tech = np.where(valid_rsi & (rsi > 70), tech - 15.0, tech)
    tech = np.where(valid_rsi & (rsi > 60) & (rsi <= 70), tech - 5.0, tech)
    valid_macd = ~_nan(macd_line) & ~_nan(macd_hist)
    tech = np.where(valid_macd & (macd_line < 0) & (macd_hist > 0), tech + 10.0, tech)
    tech = np.where(valid_macd & (macd_line > 0) & (macd_hist < 0), tech - 10.0, tech)
    valid_bb = ~_nan(bb_lower) & ~_nan(bb_upper)
    tech = np.where(valid_bb & (close < bb_lower), tech + 15.0, tech)
    tech = np.where(valid_bb & (close >= bb_lower) & (close < bb_lower * 1.02), tech + 8.0, tech)
    valid_adx = ~_nan(adx)
    tech = np.where(valid_adx & (adx < 20), tech + 5.0, tech)
    tech = np.where(valid_adx & (adx > 40), tech - 5.0, tech)
    tech = np.clip(tech, 0, 100)

    # VOLATILITY OPPORTUNITY
    vol = np.full(n, 50.0)
    valid_atr_p = ~_nan(atr_pctl)
    vol = np.where(valid_atr_p & (atr_pctl > 80), vol + 20.0, vol)
    vol = np.where(valid_atr_p & (atr_pctl > 60) & (atr_pctl <= 80), vol + 10.0, vol)
    vol = np.where(valid_atr_p & (atr_pctl < 30), vol - 10.0, vol)
    vol = np.where(drawdown < -20, vol + 30.0, vol)
    vol = np.where((drawdown >= -20) & (drawdown < -10), vol + 20.0, vol)
    vol = np.where((drawdown >= -10) & (drawdown < -5), vol + 10.0, vol)
    vol = np.where(drawdown > -1, vol - 10.0, vol)
    vol = np.clip(vol, 0, 100)

    # STATISTICAL DEVIATION
    stat = np.full(n, 50.0)
    stat = np.where(avg_z < -2.0, stat + 50.0, stat)
    stat = np.where((avg_z >= -2.0) & (avg_z < -1.5), stat + 35.0, stat)
    stat = np.where((avg_z >= -1.5) & (avg_z < -1.0), stat + 20.0, stat)
    stat = np.where((avg_z >= -1.0) & (avg_z < -0.5), stat + 10.0, stat)
    stat = np.where(avg_z > 1.5, stat - 30.0, stat)
    stat = np.where((avg_z > 1.0) & (avg_z <= 1.5), stat - 15.0, stat)
    stat = np.clip(stat, 0, 100)

    # MACRO FX (constant across all bars — uses current rate per existing behavior)
    dev_pct = (usd_inr_rate - 83.0) / 83.0 * 100
    macro = 50.0
    if dev_pct > 5:
        macro -= 20
    elif dev_pct > 2:
        macro -= 10
    elif dev_pct < -5:
        macro += 20
    elif dev_pct < -2:
        macro += 10
    macro = max(0, min(100, macro))
    macro_arr = np.full(n, macro)

    # COMPOSITE
    composite = (
        tech * weights["technical_momentum"]
        + vol * weights["volatility_opportunity"]
        + stat * weights["statistical_deviation"]
        + macro_arr * weights["macro_fx"]
    )

    return composite, atr_arr


# ════════════════════════════════════════════════════════════════
# 3. DATA PREPARATION
# ════════════════════════════════════════════════════════════════

@dataclass
class AssetData:
    """Pre-computed per-asset arrays aligned to the simulation date index."""
    symbol: str
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    score: np.ndarray
    atr: np.ndarray
    tradeable: np.ndarray   # bool — True on real trading days (not forward-filled)
    first_valid_idx: int    # index in the aligned array where data + warm-up is valid
    cost_class: str


def prepare_multi_asset_data(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    asset_meta: Dict[str, Dict[str, str]],
    usd_inr_rate: float = 83.5,
    warm_up_bars: int = 200,
    padding_days: int = 400,
    max_date: Optional[datetime] = None,
    cost_class_override: Optional[str] = None,
) -> Tuple[pd.DatetimeIndex, Dict[str, AssetData]]:
    """
    Fetch historical data for all assets, align to a common date index,
    forward-fill for MTM, compute scores + ATR, and mark warm-up periods.

    Parameters
    ----------
    symbols : list of ticker strings
    start_date, end_date : simulation window
    asset_meta : {symbol: {"asset_type": ..., "currency": ...}}
    warm_up_bars : min bars before scores are considered valid (indicator warm-up)
    padding_days : extra days before start_date to fetch for warm-up computation
    max_date : datetime, optional
        When set, truncate price data *before* score computation so that
        no information beyond this date leaks into indicator calculations.
        CPCV callers should pass the test-split end date here.
    cost_class_override : str, optional
        Force all assets to use this cost preset (e.g. "US_EQ_DIRECT").

    Returns
    -------
    dates : DatetimeIndex — the common aligned date index
    assets : dict of symbol → AssetData
    """
    from backend.data_providers import PriceProvider

    fetch_start = start_date - pd.Timedelta(days=padding_days)
    fetch_end = end_date
    if max_date is not None:
        fetch_end = min(end_date, max_date)

    raw_frames: Dict[str, pd.DataFrame] = {}

    for sym in symbols:
        meta = asset_meta.get(sym, {})
        exchange = meta.get("exchange")
        df = PriceProvider.fetch_historical_data(
            sym, period="max", exchange=exchange,
            start_date=fetch_start, end_date=fetch_end + pd.Timedelta(days=1),
        )
        if df is not None and not df.empty:
            if hasattr(df.index, "tz") and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            if max_date is not None:
                df = df.loc[df.index <= pd.Timestamp(max_date)]
            raw_frames[sym] = df
        else:
            logger.warning(f"No data for {sym} — skipping")

    if not raw_frames:
        raise ValueError("No data fetched for any asset")

    all_dates = sorted(set().union(*(df.index for df in raw_frames.values())))
    date_index = pd.DatetimeIndex(all_dates)

    assets: Dict[str, AssetData] = {}

    for sym, df in raw_frames.items():
        meta = asset_meta.get(sym, {})
        cost_class = resolve_cost_class(
            meta.get("asset_type", "equity"),
            meta.get("currency", "USD"),
            cost_class_override=cost_class_override,
        )

        aligned = df.reindex(date_index)
        tradeable = aligned["Close"].notna().values.copy()

        aligned = aligned.ffill()

        cl = aligned["Close"].values.astype(float)
        hi = aligned["High"].values.astype(float)
        lo = aligned["Low"].values.astype(float)
        op = aligned["Open"].values.astype(float)

        first_valid = np.argmax(~np.isnan(cl))
        cl[:first_valid] = cl[first_valid]
        hi[:first_valid] = hi[first_valid]
        lo[:first_valid] = lo[first_valid]
        op[:first_valid] = op[first_valid]

        scores, atr = vectorized_scores(
            cl, hi, lo, op, usd_inr_rate=usd_inr_rate,
        )

        first_valid_idx = int(first_valid) + warm_up_bars

        assets[sym] = AssetData(
            symbol=sym,
            open=op,
            high=hi,
            low=lo,
            close=cl,
            score=scores,
            atr=atr,
            tradeable=tradeable,
            first_valid_idx=first_valid_idx,
            cost_class=cost_class,
        )

    return date_index, assets


# ════════════════════════════════════════════════════════════════
# 4. EXIT FRAMEWORK
# ════════════════════════════════════════════════════════════════

@dataclass
class ExitParams:
    atr_init_mult: float = 2.0          # initial stop: entry − k1 × ATR
    atr_trail_mult: float = 2.5         # trailing stop: peak − k2 × ATR
    min_stop_pct: float = 4.0           # minimum stop distance (%)
    score_rel_mult: float = 0.4         # exit when score < entry_score × this
    score_abs_floor: float = 35.0       # AND score < this absolute level
    max_holding_days: int = 30          # forced time exit (trading days)
    min_hold_before_trail: int = 3      # don't activate trail until N days held
    use_atr_stop: bool = True           # when False, disables ATR stop entirely
    min_holding_days: int = 0           # prevents any exit within the first N days
    vol_regime_stop_widen: float = 1.5  # widen ATR mult by this factor in high-vol regimes


@dataclass
class Position:
    symbol: str
    units: float
    entry_price: float
    entry_date: date
    entry_score: float
    peak_price: float
    cost_class: str

    def days_held(self, current_date: date) -> int:
        return (current_date - self.entry_date).days


def check_exit(
    pos: Position,
    current_close: float,
    current_atr: float,
    current_score: float,
    current_date: date,
    params: ExitParams,
    *,
    high_vol_regime: bool = False,
) -> Tuple[bool, str]:
    """
    Composite exit check — OR logic (most conservative):
    Returns (should_exit, reason).

    *high_vol_regime* gates ATR stop widening when the regime detector
    reports high volatility.
    """
    days = pos.days_held(current_date)

    # Honour min_holding_days — suppress all exits during grace period
    if days < getattr(params, "min_holding_days", 0):
        pos.peak_price = max(pos.peak_price, current_close)
        return False, ""

    # Update peak
    pos.peak_price = max(pos.peak_price, current_close)

    # Guard: ATR might be NaN early on
    safe_atr = current_atr if not np.isnan(current_atr) else current_close * 0.02

    # ── Risk stop (trailing + initial) ─────────────────────────
    use_atr_stop = getattr(params, "use_atr_stop", True)
    if use_atr_stop:
        trail_mult = params.atr_trail_mult
        init_mult = params.atr_init_mult
        # Widen stops during high-volatility regimes to avoid whipsaw
        if high_vol_regime:
            widen = getattr(params, "vol_regime_stop_widen", 1.5)
            trail_mult *= widen
            init_mult *= widen

        trail_stop = pos.peak_price - trail_mult * safe_atr
        init_stop = pos.entry_price - init_mult * safe_atr
        min_stop_price = pos.entry_price * (1 - params.min_stop_pct / 100)
        stop_price = max(trail_stop, init_stop, min_stop_price)

        if days < params.min_hold_before_trail:
            stop_price = max(init_stop, min_stop_price)

        if current_close <= stop_price:
            return True, "stop"

    # ── Score-based take-profit (mean-reversion completion) ────
    score_exit = (
        current_score < pos.entry_score * params.score_rel_mult
        and current_score < params.score_abs_floor
    )
    if score_exit:
        return True, "score"

    # ── Time-based exit ────────────────────────────────────────
    if days >= params.max_holding_days:
        return True, "time"

    return False, ""


# ════════════════════════════════════════════════════════════════
# 5. SIMULATION CONFIG
# ════════════════════════════════════════════════════════════════

@dataclass
class SimConfig:
    """All knobs for a portfolio simulation run."""
    # Universe & dates
    symbols: List[str] = field(default_factory=list)
    start_date: datetime = None
    end_date: datetime = None
    initial_capital: float = 100_000.0

    # Entry
    entry_score_threshold: float = 70.0
    entry_threshold_mode: str = "fixed"   # "fixed" or "quantile"
    entry_score_quantile: float = 0.70    # when mode=quantile, use this percentile of in-sample scores
    entry_confirmation_bars: int = 1      # score must be >= threshold for N consecutive bars before entry

    # Exit
    exit_params: ExitParams = field(default_factory=ExitParams)

    # Position sizing
    max_positions: int = 10
    use_score_weighting: bool = True
    min_position_notional: float = 3_000.0
    use_volatility_scaling: bool = False   # scale allocation down by 1 + k * (ATR/price)
    volatility_scale_k: float = 0.5

    # Cost
    slippage_bps: float = 5.0
    slippage_model: str = SLIPPAGE_MODEL_FIXED
    slippage_impact_k: float = 10.0
    slippage_impact_gamma: float = 0.5
    cost_free: bool = False
    cost_class_override: Optional[str] = None  # e.g. "US_EQ_DIRECT" to force a cost preset

    # Core-satellite allocation
    min_invested_fraction: float = 0.0  # 0 = pure tactical; 0.5 = always keep 50% in equal-weight baseline

    # Execution convention
    execution: str = "next_open"   # "next_open" or "same_close"

    # Diagnostic: add Gaussian noise to scores (for stability/noise-sensitivity testing)
    score_noise_sigma: float = 0.0   # when > 0, score at t becomes score_t + N(0, sigma), clamped [0,100]

    # Benchmarks
    run_benchmarks: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbols": self.symbols,
            "start_date": str(self.start_date.date()) if self.start_date else None,
            "end_date": str(self.end_date.date()) if self.end_date else None,
            "initial_capital": self.initial_capital,
            "entry_score_threshold": self.entry_score_threshold,
            "entry_threshold_mode": getattr(self, "entry_threshold_mode", "fixed"),
            "entry_score_quantile": getattr(self, "entry_score_quantile", 0.70),
            "entry_confirmation_bars": getattr(self, "entry_confirmation_bars", 1),
            "exit_params": {
                "atr_init_mult": self.exit_params.atr_init_mult,
                "atr_trail_mult": self.exit_params.atr_trail_mult,
                "min_stop_pct": self.exit_params.min_stop_pct,
                "score_rel_mult": self.exit_params.score_rel_mult,
                "score_abs_floor": self.exit_params.score_abs_floor,
                "max_holding_days": self.exit_params.max_holding_days,
                "use_atr_stop": getattr(self.exit_params, "use_atr_stop", True),
                "min_holding_days": getattr(self.exit_params, "min_holding_days", 0),
                "vol_regime_stop_widen": getattr(self.exit_params, "vol_regime_stop_widen", 1.5),
            },
            "max_positions": self.max_positions,
            "use_score_weighting": self.use_score_weighting,
            "min_position_notional": self.min_position_notional,
            "use_volatility_scaling": getattr(self, "use_volatility_scaling", False),
            "volatility_scale_k": getattr(self, "volatility_scale_k", 0.5),
            "slippage_bps": self.slippage_bps,
            "slippage_model": getattr(self, "slippage_model", SLIPPAGE_MODEL_FIXED),
            "slippage_impact_k": getattr(self, "slippage_impact_k", 10.0),
            "slippage_impact_gamma": getattr(self, "slippage_impact_gamma", 0.5),
            "cost_free": getattr(self, "cost_free", False),
            "cost_class_override": getattr(self, "cost_class_override", None),
            "min_invested_fraction": getattr(self, "min_invested_fraction", 0.0),
        }


# ════════════════════════════════════════════════════════════════
# 6. PORTFOLIO SIMULATOR  (day-by-day state machine)
# ════════════════════════════════════════════════════════════════

@dataclass
class TradeRecord:
    date: str
    symbol: str
    side: str           # "ENTRY" or "EXIT"
    units: float
    price: float
    notional: float
    cost: float
    score: float
    exit_reason: str = ""
    pnl: float = 0.0
    holding_days: int = 0
    slippage: float = 0.0       # execution_price - raw_price (for diagnostics)
    post_trade_equity: float = 0.0  # portfolio value after this trade (for diagnostics)


@dataclass
class DailySnapshot:
    date: str
    equity: float
    cash: float
    n_positions: int
    invested_pct: float


class PortfolioSimulator:
    """
    Event-driven, daily-resolution portfolio simulator.

    Signal convention:
      - Signal generated at day T close (using day T indicators/scores)
      - Execution at day T+1 open (prevents look-ahead bias)
    """

    def __init__(self, config: SimConfig):
        self.cfg = config

    def run(
        self,
        date_index: pd.DatetimeIndex,
        assets: Dict[str, AssetData],
    ) -> Dict[str, Any]:
        """Run the full simulation. Returns results dict."""
        t0 = time.time()
        cfg = self.cfg

        # Trim date index to simulation window
        sim_mask = np.ones(len(date_index), dtype=bool)
        if cfg.start_date:
            sim_mask &= date_index >= pd.Timestamp(cfg.start_date)
        if cfg.end_date:
            sim_mask &= date_index <= pd.Timestamp(cfg.end_date)
        sim_indices = np.where(sim_mask)[0]

        if len(sim_indices) == 0:
            raise ValueError("No simulation dates in the given range")

        sim_start_idx = sim_indices[0]
        sim_end_idx = sim_indices[-1]

        # Effective entry threshold (fixed or quantile-based)
        effective_threshold = cfg.entry_score_threshold
        if getattr(cfg, "entry_threshold_mode", "fixed") == "quantile":
            all_scores = []
            for ad in assets.values():
                for t in range(sim_start_idx, min(sim_end_idx + 1, len(ad.score))):
                    s = ad.score[t]
                    if not np.isnan(s):
                        all_scores.append(float(s))
            if all_scores:
                q = getattr(cfg, "entry_score_quantile", 0.70) * 100
                effective_threshold = float(np.percentile(all_scores, q))
                warnings.append(f"Quantile threshold: {q}th percentile of in-sample scores = {effective_threshold:.1f}")

        # State
        cash = cfg.initial_capital
        positions: Dict[str, Position] = {}
        trade_log: List[TradeRecord] = []
        snapshots: List[DailySnapshot] = []
        warnings: List[str] = []

        # Pending signals (signal at T close → execute at T+1 open)
        pending_exits: List[Dict] = []
        pending_entries: List[Dict] = []

        # Entry confirmation counter: sym → consecutive bars above threshold
        confirmation_counter: Dict[str, int] = {}
        confirm_bars_needed = max(1, getattr(cfg, "entry_confirmation_bars", 1))

        slip_bps = 0.0 if cfg.cost_free else cfg.slippage_bps
        cost_free = getattr(cfg, "cost_free", False)
        ref_notional = getattr(cfg, "initial_capital", DEFAULT_REFERENCE_NOTIONAL)
        slip_model = getattr(cfg, "slippage_model", SLIPPAGE_MODEL_FIXED)
        slip_k = getattr(cfg, "slippage_impact_k", 10.0)
        slip_gamma = getattr(cfg, "slippage_impact_gamma", 0.5)
        score_noise_sigma = getattr(cfg, "score_noise_sigma", 0.0)

        # Core-satellite: keep a minimum fraction always invested
        min_invested_frac = getattr(cfg, "min_invested_fraction", 0.0)

        # Cost class override from config
        cost_class_ovr = getattr(cfg, "cost_class_override", None)

        def _noisy_score(ad: AssetData, idx: int) -> float:
            if idx >= len(ad.score):
                return np.nan
            s = float(ad.score[idx])
            if np.isnan(s):
                return s
            if score_noise_sigma > 0:
                s = float(np.clip(s + np.random.randn() * score_noise_sigma, 0.0, 100.0))
            return s

        for t in range(sim_start_idx, sim_end_idx + 1):
            dt = date_index[t].date()

            def _current_equity(c, pos_dict):
                eq = c
                for s, p in pos_dict.items():
                    eq += p.units * assets[s].close[t]
                return round(eq, 2)

            # ── 1. Execute pending signals at today's Open ─────────
            for sig in pending_exits:
                sym = sig["symbol"]
                if sym not in positions:
                    continue
                pos = positions[sym]
                ad = assets[sym]
                raw_p = float(ad.open[t])
                sell_notional = pos.units * raw_p
                exec_p = execution_price(
                    raw_p, "sell", slip_bps,
                    order_notional=sell_notional,
                    reference_notional=ref_notional,
                    slippage_model=slip_model,
                    slippage_impact_k=slip_k,
                    slippage_impact_gamma=slip_gamma,
                )
                notional = pos.units * exec_p
                cost = trade_cost(notional, pos.cost_class, "exit", cost_free=cost_free)
                proceeds = notional - cost
                pnl = proceeds - (pos.units * pos.entry_price)
                hold_days = pos.days_held(dt)
                cash += proceeds
                del positions[sym]
                trade_log.append(TradeRecord(
                    date=str(dt), symbol=sym, side="EXIT",
                    units=pos.units, price=round(exec_p, 4),
                    notional=round(notional, 2), cost=round(cost, 2),
                    score=round(sig.get("score", 0), 1),
                    exit_reason=sig["reason"],
                    pnl=round(pnl, 2), holding_days=hold_days,
                    slippage=round(exec_p - raw_p, 6),
                    post_trade_equity=_current_equity(cash, positions),
                ))

            for sig in pending_entries:
                sym = sig["symbol"]
                if sym in positions:
                    continue
                ad = assets[sym]
                alloc = sig["allocation"]
                raw_p = float(ad.open[t])
                exec_p = execution_price(
                    raw_p, "buy", slip_bps,
                    order_notional=alloc,
                    reference_notional=ref_notional,
                    slippage_model=slip_model,
                    slippage_impact_k=slip_k,
                    slippage_impact_gamma=slip_gamma,
                )
                cost = trade_cost(alloc, ad.cost_class, "entry", cost_free=cost_free)
                net_alloc = alloc - cost
                if net_alloc <= 0 or exec_p <= 0:
                    continue
                units = net_alloc / exec_p
                if alloc > cash:
                    alloc = cash
                    cost = trade_cost(alloc, ad.cost_class, "entry", cost_free=cost_free)
                    net_alloc = alloc - cost
                    if net_alloc <= 0:
                        continue
                    units = net_alloc / exec_p
                cash -= alloc
                positions[sym] = Position(
                    symbol=sym, units=units, entry_price=exec_p,
                    entry_date=dt, entry_score=sig["score"],
                    peak_price=exec_p, cost_class=ad.cost_class,
                )
                trade_log.append(TradeRecord(
                    date=str(dt), symbol=sym, side="ENTRY",
                    units=round(units, 6), price=round(exec_p, 4),
                    notional=round(alloc, 2), cost=round(cost, 2),
                    score=round(sig["score"], 1),
                    slippage=round(exec_p - raw_p, 6),
                    post_trade_equity=_current_equity(cash, positions),
                ))

            pending_exits.clear()
            pending_entries.clear()

            # ── 2. Check EXIT conditions (today's Close) ──────────
            for sym, pos in list(positions.items()):
                ad = assets[sym]
                score_t = _noisy_score(ad, t)
                should_exit, reason = check_exit(
                    pos,
                    ad.close[t],
                    ad.atr[t],
                    score_t,
                    dt,
                    cfg.exit_params,
                )
                if should_exit:
                    # Core-satellite guard: don't exit if it would bring
                    # invested fraction below the minimum threshold.
                    if min_invested_frac > 0 and len(positions) > 1:
                        equity_now = cash + sum(
                            p.units * assets[s].close[t] for s, p in positions.items()
                        )
                        invested_now = equity_now - cash
                        pos_value = pos.units * ad.close[t]
                        new_invested_frac = (invested_now - pos_value) / max(equity_now, 1.0)
                        if new_invested_frac < min_invested_frac:
                            continue
                    pending_exits.append({
                        "symbol": sym,
                        "reason": reason,
                        "score": score_t,
                    })

            # ── 3. Check ENTRY conditions (today's Close/Score) ───
            entry_candidates = []
            for sym, ad in assets.items():
                if sym in positions:
                    confirmation_counter.pop(sym, None)
                    continue
                if any(s["symbol"] == sym for s in pending_exits):
                    continue
                if t < ad.first_valid_idx:
                    continue
                if not ad.tradeable[t]:
                    continue
                score = _noisy_score(ad, t)
                if not np.isnan(score) and score >= effective_threshold:
                    confirmation_counter[sym] = confirmation_counter.get(sym, 0) + 1
                    if confirmation_counter[sym] >= confirm_bars_needed:
                        entry_candidates.append((sym, score))
                else:
                    confirmation_counter[sym] = 0

            # Rank by score descending, limit to available slots
            entry_candidates.sort(key=lambda x: x[1], reverse=True)
            available_slots = cfg.max_positions - len(positions) + len(pending_exits)
            pre_filter_count = len(entry_candidates)
            entry_candidates = entry_candidates[:max(0, available_slots)]

            # #region agent log
            if pre_filter_count > 0:
                try:
                    import json as _json2, time as _time2
                    _log_path2 = "/Users/Priyansh_Kesharwani/Downloads/Quantfi-main/.cursor/debug-9f2122.log"
                    _payload2 = _json2.dumps({"sessionId":"9f2122","hypothesisId":"H2_H3","location":"portfolio_simulator.py:entry_filter","message":"entry candidates filtered","data":{"date":str(dt),"pre_filter":pre_filter_count,"post_filter":len(entry_candidates),"available_slots":available_slots,"max_positions":cfg.max_positions,"n_positions":len(positions),"n_pending_exits":len(pending_exits),"candidates":[(s,round(sc,1)) for s,sc in entry_candidates]},"timestamp":int(_time2.time()*1000)})
                    with open(_log_path2, "a") as _f2: _f2.write(_payload2 + "\n")
                except Exception: pass
            # #endregion

            # Core-satellite: if we have no positions and min_invested_fraction > 0,
            # deploy baseline equal-weight allocation across all tradeable assets.
            if (
                min_invested_frac > 0
                and not positions
                and not entry_candidates
                and not pending_entries
            ):
                tradeable_syms = [
                    s for s, ad in assets.items()
                    if t >= ad.first_valid_idx and ad.tradeable[t]
                ]
                if tradeable_syms:
                    core_budget = cash * min_invested_frac * 0.95
                    per_asset = core_budget / len(tradeable_syms)
                    for sym in tradeable_syms:
                        if per_asset >= cfg.min_position_notional:
                            entry_candidates.append((sym, effective_threshold))

            # Size allocations
            if entry_candidates and cash > cfg.min_position_notional:
                total_score = sum(s for _, s in entry_candidates)
                # When core-satellite is active, limit tactical overlay to
                # the non-core fraction so the core stays invested.
                if min_invested_frac > 0:
                    allocatable = cash * 0.95
                else:
                    allocatable = cash * 0.95
                raw_weights = []
                for sym, score in entry_candidates:
                    if cfg.use_score_weighting and total_score > 0:
                        w = score / total_score
                    else:
                        w = 1.0 / len(entry_candidates)
                    if getattr(cfg, "use_volatility_scaling", False):
                        ad = assets[sym]
                        if t < len(ad.atr) and t < len(ad.close) and ad.close[t] > 0 and ad.atr[t] >= 0:
                            atr_pct = ad.atr[t] / ad.close[t]
                            k = getattr(cfg, "volatility_scale_k", 0.5)
                            w = w / (1.0 + k * atr_pct)
                    raw_weights.append((sym, score, w))
                total_w = sum(w for _, _, w in raw_weights)
                if total_w <= 0:
                    total_w = 1.0
                for sym, score, w in raw_weights:
                    alloc = allocatable * (w / total_w)
                    alloc = min(alloc, cash - cfg.min_position_notional * 0.1)
                    if alloc < cfg.min_position_notional:
                        continue
                    pending_entries.append({
                        "symbol": sym,
                        "score": score,
                        "allocation": alloc,
                    })

            # ── 4. Mark-to-market ─────────────────────────────────
            equity = cash
            for sym, pos in positions.items():
                equity += pos.units * assets[sym].close[t]

            n_pos = len(positions)
            invested_pct = ((equity - cash) / equity * 100) if equity > 0 else 0

            snapshots.append(DailySnapshot(
                date=str(dt),
                equity=round(equity, 2),
                cash=round(cash, 2),
                n_positions=n_pos,
                invested_pct=round(invested_pct, 1),
            ))

        # ── Finalize: force-close any remaining positions at last close ──
        final_t = sim_end_idx
        final_dt = date_index[final_t].date()
        for sym, pos in list(positions.items()):
            ad = assets[sym]
            raw_p = float(ad.close[final_t])
            sell_notional_f = pos.units * raw_p
            exec_p = execution_price(
                raw_p, "sell", slip_bps,
                order_notional=sell_notional_f,
                reference_notional=ref_notional,
                slippage_model=slip_model,
                slippage_impact_k=slip_k,
                slippage_impact_gamma=slip_gamma,
            )
            notional = pos.units * exec_p
            cost = trade_cost(notional, pos.cost_class, "exit", cost_free=cost_free)
            proceeds = notional - cost
            pnl = proceeds - (pos.units * pos.entry_price)
            cash += proceeds
            del positions[sym]
            trade_log.append(TradeRecord(
                date=str(final_dt), symbol=sym, side="EXIT",
                units=pos.units, price=round(exec_p, 4),
                notional=round(notional, 2), cost=round(cost, 2),
                score=round(ad.score[final_t], 1),
                exit_reason="end_of_sim",
                pnl=round(pnl, 2),
                holding_days=pos.days_held(final_dt),
                slippage=round(exec_p - raw_p, 6),
                post_trade_equity=round(cash + sum(
                    positions[s].units * assets[s].close[final_t]
                    for s in positions
                ), 2),
            ))
        positions.clear()

        elapsed = round(time.time() - t0, 2)

        # #region agent log
        try:
            import json as _json3, time as _time3
            _log_path3 = "/Users/Priyansh_Kesharwani/Downloads/Quantfi-main/.cursor/debug-9f2122.log"
            _total_entries = sum(1 for t in trade_log if t.side == "ENTRY")
            _total_exits = sum(1 for t in trade_log if t.side == "EXIT")
            _payload3 = _json3.dumps({"sessionId":"9f2122","hypothesisId":"H2_summary","location":"portfolio_simulator.py:sim_end","message":"simulation complete","data":{"max_positions":cfg.max_positions,"total_entries":_total_entries,"total_exits":_total_exits,"entry_threshold":cfg.entry_score_threshold,"confirm_bars":getattr(cfg,'entry_confirmation_bars',1),"n_snapshots":len(snapshots),"elapsed":elapsed},"timestamp":int(_time3.time()*1000)})
            with open(_log_path3, "a") as _f3: _f3.write(_payload3 + "\n")
        except Exception: pass
        # #endregion

        # ── Compute analytics ─────────────────────────────────────
        result = self._compute_analytics(
            snapshots, trade_log, cfg, date_index, assets,
            sim_start_idx, sim_end_idx, elapsed, warnings,
        )
        return result

    def _compute_analytics(
        self,
        snapshots: List[DailySnapshot],
        trade_log: List[TradeRecord],
        cfg: SimConfig,
        date_index: pd.DatetimeIndex,
        assets: Dict[str, AssetData],
        sim_start_idx: int,
        sim_end_idx: int,
        elapsed: float,
        warnings: List[str],
    ) -> Dict[str, Any]:
        """Compute portfolio-level and per-asset metrics."""
        equities = np.array([s.equity for s in snapshots])
        dates_str = [s.date for s in snapshots]

        if len(equities) < 2:
            warnings.append("Insufficient data for analytics")
            return {"warnings": warnings, "equity_curve": [], "trades": []}

        # Returns
        daily_returns = np.diff(equities) / equities[:-1]
        daily_returns = daily_returns[~np.isnan(daily_returns)]

        total_return_pct = (equities[-1] / equities[0] - 1) * 100
        n_days = len(equities)
        n_years = n_days / 252
        cagr = ((equities[-1] / equities[0]) ** (1 / max(n_years, 0.01)) - 1) * 100

        # Sharpe (annualised, excess over 0 for simplicity)
        if daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Sortino (downside deviation)
        neg_returns = daily_returns[daily_returns < 0]
        if len(neg_returns) > 0 and neg_returns.std() > 0:
            sortino = (daily_returns.mean() / neg_returns.std()) * np.sqrt(252)
        else:
            sortino = 0.0

        # Max drawdown
        running_max = np.maximum.accumulate(equities)
        drawdowns = (equities - running_max) / running_max * 100
        max_dd = float(np.min(drawdowns))

        # Calmar
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

        # Trade stats
        exits = [t for t in trade_log if t.side == "EXIT"]
        total_trades = len(exits)
        wins = [t for t in exits if t.pnl > 0]
        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
        avg_holding = np.mean([t.holding_days for t in exits]) if exits else 0
        total_costs = sum(t.cost for t in trade_log)
        cost_drag = total_costs / cfg.initial_capital * 100

        # Time in market
        invested_pcts = [s.invested_pct for s in snapshots]
        time_in_market = np.mean(invested_pcts) if invested_pcts else 0

        # Exit reason breakdown
        exit_reasons = {}
        for t in exits:
            r = t.exit_reason or "unknown"
            exit_reasons[r] = exit_reasons.get(r, 0) + 1

        # Per-asset breakdown
        asset_breakdown = {}
        for sym in assets:
            sym_trades = [t for t in exits if t.symbol == sym]
            sym_entries = [t for t in trade_log if t.symbol == sym and t.side == "ENTRY"]
            if sym_trades:
                asset_breakdown[sym] = {
                    "trades": len(sym_trades),
                    "total_pnl": round(sum(t.pnl for t in sym_trades), 2),
                    "win_rate": round(
                        len([t for t in sym_trades if t.pnl > 0]) / len(sym_trades) * 100, 1
                    ),
                    "avg_holding_days": round(np.mean([t.holding_days for t in sym_trades]), 1),
                    "total_costs": round(
                        sum(t.cost for t in sym_trades) + sum(t.cost for t in sym_entries), 2
                    ),
                }

        # ── Benchmarks ────────────────────────────────────────────
        benchmarks = {}
        if cfg.run_benchmarks:
            benchmarks = self._run_benchmarks(
                date_index, assets, sim_start_idx, sim_end_idx, cfg.initial_capital,
            )

        # ── Build result ──────────────────────────────────────────
        # Thin equity curve for frontend (max 500 points)
        ec = [{"date": s.date, "equity": s.equity, "cash": s.cash,
               "n_positions": s.n_positions, "invested_pct": s.invested_pct}
              for s in snapshots]
        if len(ec) > 500:
            step = max(1, len(ec) // 500)
            ec = ec[::step]
            if ec[-1]["date"] != snapshots[-1].date:
                ec.append({"date": snapshots[-1].date, "equity": snapshots[-1].equity,
                           "cash": snapshots[-1].cash, "n_positions": snapshots[-1].n_positions,
                           "invested_pct": snapshots[-1].invested_pct})

        trades_out = [
            {
                "date": t.date, "symbol": t.symbol, "side": t.side,
                "units": t.units, "price": t.price, "notional": t.notional,
                "cost": t.cost, "score": t.score,
                "exit_reason": t.exit_reason, "pnl": t.pnl,
                "holding_days": t.holding_days,
                "slippage": getattr(t, "slippage", 0.0),
                "post_trade_equity": getattr(t, "post_trade_equity", 0.0),
            }
            for t in trade_log
        ]

        return {
            "config": cfg.to_dict(),
            "total_return_pct": round(total_return_pct, 2),
            "cagr_pct": round(cagr, 2),
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "calmar_ratio": round(calmar, 3),
            "max_drawdown_pct": round(max_dd, 2),
            "total_trades": total_trades,
            "win_rate": round(win_rate, 1),
            "avg_holding_days": round(avg_holding, 1),
            "time_in_market_pct": round(time_in_market, 1),
            "total_costs": round(total_costs, 2),
            "cost_drag_pct": round(cost_drag, 2),
            "exit_reasons": exit_reasons,
            "asset_breakdown": asset_breakdown,
            "equity_curve": ec,
            "trades": trades_out,
            "benchmarks": benchmarks,
            "data_range": f"{snapshots[0].date} → {snapshots[-1].date}" if snapshots else "",
            "computation_time_s": elapsed,
            "warnings": warnings,
        }

    def _run_benchmarks(
        self,
        date_index: pd.DatetimeIndex,
        assets: Dict[str, AssetData],
        start_idx: int,
        end_idx: int,
        initial_capital: float,
    ) -> Dict[str, Any]:
        """Compute buy-and-hold and uniform periodic benchmarks."""
        n = end_idx - start_idx + 1
        syms = list(assets.keys())
        n_assets = len(syms)
        if n_assets == 0 or n < 2:
            return {}

        # ── Buy-and-Hold (equal-weight at start) ──────────────────
        bnh_alloc = initial_capital / n_assets
        bnh_units = {}
        for sym in syms:
            ad = assets[sym]
            price = ad.close[start_idx]
            if price > 0 and not np.isnan(price):
                bnh_units[sym] = bnh_alloc / price
            else:
                bnh_units[sym] = 0

        bnh_equity = []
        for t in range(start_idx, end_idx + 1):
            val = sum(
                bnh_units[sym] * assets[sym].close[t]
                for sym in syms
            )
            bnh_equity.append(val)

        bnh_equity = np.array(bnh_equity)
        bnh_return = (bnh_equity[-1] / bnh_equity[0] - 1) * 100 if bnh_equity[0] > 0 else 0
        n_years = n / 252
        bnh_cagr = ((bnh_equity[-1] / bnh_equity[0]) ** (1 / max(n_years, 0.01)) - 1) * 100 if bnh_equity[0] > 0 else 0
        bnh_rets = np.diff(bnh_equity) / bnh_equity[:-1]
        bnh_sharpe = (bnh_rets.mean() / bnh_rets.std() * np.sqrt(252)) if bnh_rets.std() > 0 else 0
        bnh_rm = np.maximum.accumulate(bnh_equity)
        bnh_dd = float(np.min((bnh_equity - bnh_rm) / bnh_rm * 100))

        # ── Uniform Periodic (monthly equal investment) ───────────
        unif_cash = initial_capital
        unif_units = {s: 0.0 for s in syms}
        unif_invested = 0.0
        monthly_budget = initial_capital / max(n_years * 12, 1)
        last_invest_month = -1

        unif_equity = []
        for i, t in enumerate(range(start_idx, end_idx + 1)):
            dt = date_index[t]
            month_key = dt.year * 12 + dt.month
            if month_key != last_invest_month and unif_cash >= monthly_budget:
                per_asset = monthly_budget / n_assets
                for sym in syms:
                    price = assets[sym].close[t]
                    if price > 0 and not np.isnan(price):
                        units = per_asset / price
                        unif_units[sym] += units
                unif_cash -= monthly_budget
                unif_invested += monthly_budget
                last_invest_month = month_key

            val = unif_cash + sum(
                unif_units[sym] * assets[sym].close[t]
                for sym in syms
            )
            unif_equity.append(val)

        unif_equity = np.array(unif_equity)
        unif_return = (unif_equity[-1] / unif_equity[0] - 1) * 100 if unif_equity[0] > 0 else 0
        unif_cagr = ((unif_equity[-1] / unif_equity[0]) ** (1 / max(n_years, 0.01)) - 1) * 100 if unif_equity[0] > 0 else 0

        # Per-asset price curves normalised to initial_capital base
        # so they can be overlaid on the equity chart at comparable scale
        asset_price_curves: Dict[str, List[Dict[str, Any]]] = {}
        for sym in syms:
            ad = assets[sym]
            base_price = ad.close[start_idx]
            if base_price <= 0 or np.isnan(base_price):
                continue
            scale = initial_capital / base_price
            curve = []
            step_a = max(1, n // 500)
            for i in range(0, n, step_a):
                t = start_idx + i
                d = str(date_index[t].date())
                curve.append({"date": d, "value": round(float(ad.close[t]) * scale, 2)})
            asset_price_curves[sym] = curve

        # Thin benchmark curves
        bnh_curve = []
        unif_curve = []
        step = max(1, n // 500)
        for i in range(0, n, step):
            t = start_idx + i
            d = str(date_index[t].date())
            bnh_curve.append({"date": d, "equity": round(bnh_equity[i], 2)})
            unif_curve.append({"date": d, "equity": round(unif_equity[i], 2)})

        return {
            "buy_and_hold": {
                "total_return_pct": round(bnh_return, 2),
                "cagr_pct": round(bnh_cagr, 2),
                "sharpe_ratio": round(float(bnh_sharpe), 3),
                "max_drawdown_pct": round(bnh_dd, 2),
                "equity_curve": bnh_curve,
            },
            "uniform_periodic": {
                "total_return_pct": round(unif_return, 2),
                "cagr_pct": round(unif_cagr, 2),
                "equity_curve": unif_curve,
            },
            "asset_prices": asset_price_curves,
        }


def export_trades_csv(
    csv_path: str,
    trade_log: Optional[List[TradeRecord]] = None,
    trades: Optional[List[Dict[str, Any]]] = None,
    run_id: str = "",
) -> str:
    """Write every executed trade to a CSV for diagnostics.

    Provide either trade_log (List[TradeRecord]) or trades (list of dicts from result).
    Columns: date, symbol, side, units, price, notional, cost, slippage,
    post_trade_equity, score, exit_reason, pnl, holding_days.
    Creates parent directory if needed. Returns the absolute path written.
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "date", "symbol", "side", "units", "price", "notional", "cost",
        "slippage", "post_trade_equity", "score", "exit_reason", "pnl", "holding_days",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        if trade_log is not None:
            for t in trade_log:
                row = {
                    "date": t.date,
                    "symbol": t.symbol,
                    "side": t.side,
                    "units": t.units,
                    "price": t.price,
                    "notional": t.notional,
                    "cost": t.cost,
                    "slippage": getattr(t, "slippage", 0.0),
                    "post_trade_equity": getattr(t, "post_trade_equity", 0.0),
                    "score": t.score,
                    "exit_reason": t.exit_reason or "",
                    "pnl": t.pnl,
                    "holding_days": t.holding_days,
                }
                w.writerow(row)
        elif trades is not None:
            for row in trades:
                w.writerow({k: row.get(k, "") for k in fieldnames})
        else:
            raise ValueError("Provide either trade_log or trades")
    return str(path.resolve())
