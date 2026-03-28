"""CryptoDirectionalScorer: self-calibrating, IC-EWMA weighted scoring system."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EPS = 1e-12

@dataclass
class ScoringConfig:
    rsi_period: int = 14
    rsi_oversold: float = 25.0
    rsi_overbought: float = 75.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    adx_period: int = 14
    compression_window: int = 252
    z_short_weight: float = 0.40
    z_mid_weight: float = 0.35
    vol_threshold: float = 2.0
    vol_scale: float = 0.5
    vol_filter_pctl: float = 0.90
    vol_filter_floor: float = 0.3
    vol_filter_width: float = 0.10
    funding_window: int = 504
    entry_threshold: float = 40.0
    exit_threshold: float = 15.0
    trend_ma_fast: int = 20
    trend_ma_slow: int = 50
    ic_window: int = 252
    ic_horizon: int = 20
    ic_alpha: float = 5.0
    ic_shrink: float = 0.2

def _sigmoid(x: np.ndarray, threshold: float, scale: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(x - threshold) / (scale + EPS)))

def _self_calibrating_tanh(
    signal: pd.Series,
    window: int,
    scale_factor: float = 100.0,
) -> pd.Series:
    """Tanh compression using rolling std for auto-normalization."""
    rolling_std = signal.rolling(window, min_periods=max(2, window // 4)).std().fillna(signal.expanding().std())
    rolling_std = rolling_std.clip(lower=EPS)
    return np.tanh(signal / rolling_std) * scale_factor

class CryptoDirectionalScorer:
    """Computes directional score in [-100, +100].

    Positive = long bias, Negative = short bias.
    Uses IC-EWMA adaptive weights from weights/ic_ewma.py.
    """

    COMPONENT_NAMES = [
        "rsi", "macd", "bollinger", "trend_ma", "adx",
        "zscore", "funding", "volume_momentum", "oi_change", "vol_filter_score",
    ]

    def __init__(self, config: Optional[ScoringConfig] = None):
        self.config = config or ScoringConfig()
        self._ic_ewma = None
        self._n_components = len(self.COMPONENT_NAMES)

    def _get_ic_ewma(self):
        """Lazy init of IC-EWMA so weights/ic_ewma.py import is deferred."""
        if self._ic_ewma is None:
            from engine.weights.ic_ewma import IC_EWMA_Weights
            self._ic_ewma = IC_EWMA_Weights(
                ic_window=self.config.ic_window,
                ic_forward_horizon=self.config.ic_horizon,
                alpha=self.config.ic_alpha,
                lambda_shrink=self.config.ic_shrink,
                mode="offline",
            )
        return self._ic_ewma

    def compute(
        self,
        ohlcv: pd.DataFrame,
        funding_rates: Optional[pd.Series] = None,
        open_interest: Optional[pd.Series] = None,
    ) -> pd.Series:
        """IC-EWMA weighted scores — OFFLINE EVALUATION ONLY.

        WARNING: This method uses forward returns (.shift(-horizon)) to compute
        IC weights.  It MUST NOT be used in backtests or live trading because
        it introduces look-ahead bias.  Use compute_with_uniform_weights()
        for any path where future data is unavailable.
        """
        components = self._compute_all_components(ohlcv, funding_rates, open_interest)

        ic_ewma = self._get_ic_ewma()
        forward_ret = ohlcv["close"].pct_change(self.config.ic_horizon).shift(-self.config.ic_horizon)
        weights, meta = ic_ewma.update(components.values, forward_ret.values)

        raw_scores = (components * weights).sum(axis=1)

        vol_filter = self._compute_vol_filter(ohlcv)
        return raw_scores * vol_filter

    def compute_with_uniform_weights(
        self,
        ohlcv: pd.DataFrame,
        funding_rates: Optional[pd.Series] = None,
        open_interest: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Compute scores with uniform weights, self-calibrated to fill [-100, 100].

        Uses only active components (non-zero variance) and normalizes the
        composite to a target scale so entry/exit thresholds remain meaningful
        regardless of how many components are live.
        """
        components = self._compute_all_components(ohlcv, funding_rates, open_interest)

        warmup_end = min(self.config.compression_window, len(components))
        col_std = components.iloc[:warmup_end].std()
        active_cols = col_std[col_std > 1.0].index.tolist()
        if not active_cols:
            return pd.Series(0.0, index=ohlcv.index)

        active = components[active_cols]
        raw = active.mean(axis=1)

        target_scale = 35.0
        window = self.config.compression_window
        rolling_std = raw.rolling(window, min_periods=20).std()
        rolling_std = rolling_std.fillna(
            raw.expanding(min_periods=5).std()
        ).clip(lower=EPS)
        normalized = (raw / rolling_std * target_scale).clip(-100, 100)

        vol_filter = self._compute_vol_filter(ohlcv).fillna(1.0)
        result = (normalized * vol_filter).fillna(0.0)
        return result

    def _compute_all_components(
        self,
        ohlcv: pd.DataFrame,
        funding_rates: Optional[pd.Series],
        open_interest: Optional[pd.Series],
    ) -> pd.DataFrame:
        """Compute all 10 scoring components. Returns DataFrame (T, 10)."""
        c = self.config
        close = ohlcv["close"]
        high = ohlcv.get("high", close)
        low = ohlcv.get("low", close)
        volume = ohlcv.get("volume", pd.Series(0.0, index=ohlcv.index))
        open_ = ohlcv.get("open", close)

        components = pd.DataFrame(index=ohlcv.index)

        components["rsi"] = self._score_rsi(close, c.rsi_period, c.rsi_oversold, c.rsi_overbought)
        components["macd"] = self._score_macd(close, c.macd_fast, c.macd_slow, c.macd_signal, high, low, c.atr_period, c.compression_window)
        components["bollinger"] = self._score_bollinger(close, c.bb_period, c.bb_std, c.compression_window)
        components["trend_ma"] = self._score_trend_ma(close, c.trend_ma_fast, c.trend_ma_slow, high, low, c.atr_period, c.compression_window)
        components["adx"] = self._score_adx(high, low, close, c.adx_period, c.compression_window)
        components["zscore"] = self._score_zscore(close, c.z_short_weight, c.z_mid_weight, c.compression_window)
        components["funding"] = self._score_funding(funding_rates, c.funding_window, c.compression_window)
        components["volume_momentum"] = self._score_volume_momentum(volume, open_, close, c.vol_threshold, c.vol_scale)
        components["oi_change"] = self._score_oi_change(open_interest, c.compression_window)
        components["vol_filter_score"] = self._score_vol_filter_contribution(close, high, low, c.atr_period)

        return components.fillna(0.0)

    def _score_rsi(self, close: pd.Series, period: int, oversold: float, overbought: float) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
        rs = gain / (loss + EPS)
        rsi = 100.0 - 100.0 / (1.0 + rs)

        score = pd.Series(0.0, index=close.index)
        below = rsi < oversold
        above = rsi > overbought
        score[below] = (oversold - rsi[below]) * (100.0 / oversold)
        score[above] = (overbought - rsi[above]) * (100.0 / (100.0 - overbought))
        return score

    def _score_macd(self, close, fast, slow, signal, high, low, atr_period, comp_window) -> pd.Series:
        ema_f = close.ewm(span=fast, min_periods=1).mean()
        ema_s = close.ewm(span=slow, min_periods=1).mean()
        macd_line = ema_f - ema_s
        signal_line = macd_line.ewm(span=signal, min_periods=1).mean()
        histogram = macd_line - signal_line

        atr = self._compute_atr_series(high, low, close, atr_period)
        macd_norm = histogram / (atr + EPS)
        return _self_calibrating_tanh(macd_norm, comp_window)

    def _score_bollinger(self, close, period, std_mult, comp_window) -> pd.Series:
        ma = close.rolling(period, min_periods=1).mean()
        std = close.rolling(period, min_periods=1).std().fillna(EPS)
        bb_pct = (close - ma) / (std * std_mult + EPS)
        return _self_calibrating_tanh(-bb_pct, comp_window)

    def _score_trend_ma(self, close, fast, slow, high, low, atr_period, comp_window) -> pd.Series:
        ma_f = close.rolling(fast, min_periods=1).mean()
        ma_s = close.rolling(slow, min_periods=1).mean()
        atr = self._compute_atr_series(high, low, close, atr_period)
        spread = (ma_f - ma_s) / (atr + EPS)
        return _self_calibrating_tanh(spread, comp_window)

    def _score_adx(self, high, low, close, period, comp_window) -> pd.Series:
        """ADX as a trend strength filter (0-100 range, then normalized)."""
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        atr = self._compute_atr_series(high, low, close, period)
        plus_di = (plus_dm.rolling(period, min_periods=1).mean() / (atr + EPS)) * 100
        minus_di = (minus_dm.rolling(period, min_periods=1).mean() / (atr + EPS)) * 100

        dx = (abs(plus_di - minus_di) / (plus_di + minus_di + EPS)) * 100
        adx = dx.rolling(period, min_periods=1).mean()

        direction = np.sign(plus_di - minus_di)
        adx_signal = (adx - 25.0) / 25.0 * direction
        return _self_calibrating_tanh(adx_signal, comp_window)

    def _score_zscore(self, close, w_short, w_mid, comp_window) -> pd.Series:
        z20 = (close - close.rolling(20, min_periods=1).mean()) / (close.rolling(20, min_periods=1).std() + EPS)
        z50 = (close - close.rolling(50, min_periods=1).mean()) / (close.rolling(50, min_periods=1).std() + EPS)
        z100 = (close - close.rolling(100, min_periods=1).mean()) / (close.rolling(100, min_periods=1).std() + EPS)

        w_long = max(0.0, 1.0 - w_short - w_mid)
        z_blend = w_short * z20 + w_mid * z50 + w_long * z100
        return _self_calibrating_tanh(-z_blend, comp_window)

    def _score_funding(self, funding_rates: Optional[pd.Series], window: int, comp_window: int) -> pd.Series:
        if funding_rates is None or funding_rates.empty:
            return pd.Series(0.0, name="funding")
        mu = funding_rates.rolling(window, min_periods=1).mean()
        sigma = funding_rates.rolling(window, min_periods=1).std().fillna(EPS)
        funding_z = (funding_rates - mu) / (sigma + EPS)
        return _self_calibrating_tanh(-funding_z, comp_window)

    def _score_volume_momentum(self, volume, open_, close, threshold, scale) -> pd.Series:
        vol_mean = volume.rolling(20, min_periods=1).mean()
        vol_ratio = volume / (vol_mean + EPS)
        activation = pd.Series(
            _sigmoid(vol_ratio.values, threshold, scale),
            index=volume.index,
        )
        direction = np.sign(close - open_)
        return activation * direction * 100.0

    def _score_oi_change(self, open_interest: Optional[pd.Series], comp_window: int) -> pd.Series:
        if open_interest is None or open_interest.empty:
            return pd.Series(0.0, name="oi_change")
        oi_pct = open_interest.pct_change().fillna(0.0)
        return _self_calibrating_tanh(oi_pct, comp_window)

    def _score_vol_filter_contribution(self, close, high, low, atr_period) -> pd.Series:
        """Volatility awareness score (not the filter itself, just contribution signal)."""
        atr = self._compute_atr_series(high, low, close, atr_period)
        atr_pct = atr / (close + EPS)
        z = (atr_pct - atr_pct.rolling(100, min_periods=1).mean()) / (atr_pct.rolling(100, min_periods=1).std() + EPS)
        return -z.clip(-3, 3) * 33.33

    def _compute_vol_filter(self, ohlcv: pd.DataFrame) -> pd.Series:
        """ATR percentile-based dampening of scores in high-volatility environments."""
        c = self.config
        close = ohlcv["close"]
        high = ohlcv.get("high", close)
        low = ohlcv.get("low", close)
        atr = self._compute_atr_series(high, low, close, c.atr_period)
        atr_pctl = atr.rolling(252, min_periods=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        ).fillna(0.5)

        filter_val = (1.0 - (atr_pctl - c.vol_filter_pctl) / (c.vol_filter_width + EPS)).clip(
            c.vol_filter_floor, 1.0
        )
        return filter_val

    @staticmethod
    def _compute_atr_series(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        prev_close = close.shift(1).fillna(close)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        return tr.rolling(period, min_periods=1).mean()

def verify_score_reachability(
    scores: pd.Series,
    entry_threshold: float,
) -> Dict:
    """Check that entry_threshold is achievable given score distribution."""
    valid_scores = scores.dropna()
    if len(valid_scores) == 0:
        return {"mean": 0, "std": 0, "p5": 0, "p95": 0, "entry_pct": 0, "ok": False}
    entry_pct = float((valid_scores.abs() > entry_threshold).mean() * 100)
    result = {
        "mean": float(valid_scores.mean()),
        "std": float(valid_scores.std()),
        "p5": float(valid_scores.quantile(0.05)),
        "p95": float(valid_scores.quantile(0.95)),
        "entry_pct": entry_pct,
        "ok": 0.5 <= entry_pct <= 50.0,
    }
    return result
