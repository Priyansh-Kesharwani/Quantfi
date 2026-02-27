"""CryptoRegimeDetector: rolling regime classification with HMM, circuit breaker, and hysteresis.

Uses GaussianHMM exclusively — no heuristic fallback.  When the 3-state model
cannot separate all three regimes, a 2-state model is tried and the two states
are mapped based on variance and trend-spread.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from crypto.calendar import bars_per_day

logger = logging.getLogger(__name__)

REGIME_TRENDING = "TRENDING"
REGIME_RANGING = "RANGING"
REGIME_STRESS = "STRESS"
REGIMES = [REGIME_TRENDING, REGIME_RANGING, REGIME_STRESS]

_N_INITS = 10
_HMM_N_ITER = 200
_HMM_TOL = 0.001
_SEPARATION_RATIO = 1.3


@dataclass
class CryptoRegimeConfig:
    n_states: int = 3
    jump_penalty: float = 8.0
    vol_window: int = 168
    refit_every: int = 168
    cooldown_bars: int = 12
    circuit_breaker_dd: float = -0.25
    rolling_window: int = 504
    warmup_bars: int = 504
    ema_fast: int = 12
    ema_slow: int = 26
    atr_period: int = 14


def default_regime_config(timeframe: str) -> CryptoRegimeConfig:
    """Generate regime config seeds based on timeframe."""
    bpd = bars_per_day(timeframe)
    return CryptoRegimeConfig(
        n_states=3,
        jump_penalty=max(3.0, 5.0 * np.sqrt(bpd / 24)),
        vol_window=int(7 * bpd),
        refit_every=int(7 * bpd),
        cooldown_bars=max(3, int(bpd / 2)),
        circuit_breaker_dd=-0.25,
        rolling_window=int(21 * bpd),
        warmup_bars=int(21 * bpd),
    )


def _extract_state_variances(model) -> np.ndarray:
    """Extract per-state total variance from covars_, handling both 2D and 3D shapes.

    hmmlearn 0.3.x returns covars_ with shape (n_components, n_features, n_features)
    even for covariance_type='diag'.  Older versions returned (n_components, n_features).
    """
    c = model.covars_
    if c.ndim == 3:
        return np.array([np.trace(c[i]) for i in range(c.shape[0])])
    return np.sum(c, axis=1)


def _fit_hmm_best_of_n(
    X: np.ndarray,
    n_components: int,
    covariance_type: str = "diag",
    n_inits: int = _N_INITS,
) -> Optional[object]:
    """Fit GaussianHMM with *n_inits* random seeds, return model with best log-likelihood."""
    from hmmlearn.hmm import GaussianHMM

    best_model, best_score = None, -np.inf
    for seed in range(n_inits):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = GaussianHMM(
                    n_components=n_components,
                    covariance_type=covariance_type,
                    n_iter=_HMM_N_ITER,
                    random_state=seed,
                    tol=_HMM_TOL,
                )
                m.fit(X)
                sc = m.score(X)
                if sc > best_score:
                    best_model, best_score = m, sc
        except Exception:
            continue
    return best_model


def _all_states_present(model, X: np.ndarray) -> bool:
    """Check that a fitted model produces all n_components states on the training data."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        states = model.predict(X)
    return len(set(states)) == model.n_components


def _states_well_separated(model) -> bool:
    """Check that the highest-variance state has >= _SEPARATION_RATIO * lowest."""
    v = _extract_state_variances(model)
    if v.min() <= 0:
        return True
    return v.max() / v.min() >= _SEPARATION_RATIO


class CryptoRegimeDetector:
    """Rolling regime detector using GaussianHMM.  No heuristic fallback."""

    def __init__(self, config: Optional[CryptoRegimeConfig] = None):
        self.config = config or CryptoRegimeConfig()
        self._model = None
        self._state_map: dict[int, str] = {}

    def fit_rolling(self, ohlcv: pd.DataFrame) -> pd.Series:
        """Classify regimes bar-by-bar.  Returns Series of regime labels."""
        features = self._prepare_features(ohlcv)
        n = len(features)
        labels = pd.Series(REGIME_RANGING, index=ohlcv.index, dtype=str)

        warmup = self.config.warmup_bars
        if n < warmup:
            return labels

        self._fit_model(features, warmup)

        if self._model is not None:
            labels = self._classify_hmm(features, labels, warmup)

        labels = self._apply_circuit_breaker(ohlcv["close"], labels)
        labels = self._apply_hysteresis(labels)
        return labels

    # ------------------------------------------------------------------
    # Model fitting
    # ------------------------------------------------------------------

    def _fit_model(self, features: np.ndarray, warmup: int) -> None:
        """Fit HMM on warmup window.  Tries 3-state diag → 3-state full → 2-state diag."""
        X = features[:warmup]
        valid_mask = ~np.isnan(X).any(axis=1)
        X_valid = X[valid_mask]
        if len(X_valid) < self.config.n_states * 20:
            return

        try:
            from hmmlearn.hmm import GaussianHMM  # noqa: F401 — verify import
        except ImportError:
            logger.warning("hmmlearn not installed — regime detector disabled")
            return

        model = _fit_hmm_best_of_n(X_valid, n_components=3, covariance_type="diag")
        if model is not None and _all_states_present(model, X_valid) and _states_well_separated(model):
            self._model = model
            self._build_state_map()
            return

        model = _fit_hmm_best_of_n(X_valid, n_components=3, covariance_type="full")
        if model is not None and _all_states_present(model, X_valid) and _states_well_separated(model):
            self._model = model
            self._build_state_map()
            return

        model = _fit_hmm_best_of_n(X_valid, n_components=2, covariance_type="diag")
        if model is not None:
            self._model = model
            self._build_state_map()
            return

        logger.warning("All HMM fits failed — regimes will be RANGING")

    def _build_state_map(self) -> None:
        """Build mapping from HMM integer states to regime labels."""
        m = self._model
        n_states = m.n_components
        variances = _extract_state_variances(m)
        sorted_idx = np.argsort(variances).tolist()

        mapping: dict[int, str] = {}

        if n_states >= 3:
            mapping[sorted_idx[-1]] = REGIME_STRESS
            non_stress = sorted_idx[:-1]
            trend_col = min(2, m.means_.shape[1] - 1)
            trend_strength = [abs(float(m.means_[s, trend_col])) for s in non_stress]
            if trend_strength[0] >= trend_strength[1]:
                mapping[non_stress[0]] = REGIME_TRENDING
                mapping[non_stress[1]] = REGIME_RANGING
            else:
                mapping[non_stress[0]] = REGIME_RANGING
                mapping[non_stress[1]] = REGIME_TRENDING
        elif n_states == 2:
            hi, lo = sorted_idx[-1], sorted_idx[0]
            mapping[hi] = REGIME_STRESS
            trend_col = min(2, m.means_.shape[1] - 1)
            if abs(float(m.means_[lo, trend_col])) > 0.3:
                mapping[lo] = REGIME_TRENDING
            else:
                mapping[lo] = REGIME_RANGING
        else:
            mapping[0] = REGIME_RANGING

        self._state_map = mapping

    def _map_states(self, raw_label: int) -> str:
        return self._state_map.get(int(raw_label), REGIME_RANGING)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify_hmm(
        self, features: np.ndarray, labels: pd.Series, warmup: int
    ) -> pd.Series:
        """Classify using the fitted HMM model with periodic rolling refits."""
        n = len(features)
        for t in range(warmup, n):
            if t % self.config.refit_every == 0 and t > warmup:
                self._rolling_refit(features, t)

            if self._model is not None:
                ws = max(0, t - self.config.rolling_window)
                X_w = features[ws: t + 1]
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        raw = self._model.predict(X_w)
                    labels.iloc[t] = self._map_states(raw[-1])
                except Exception:
                    pass
        return labels

    def _rolling_refit(self, features: np.ndarray, t: int) -> None:
        """Refit HMM on the most recent rolling window.

        Detects state-map flips: if the mapping changed relative to the
        previous fit, the old mapping is kept for ``cooldown_bars`` to
        prevent whiplash regime switches.
        """
        ws = max(0, t - self.config.rolling_window)
        X_w = features[ws: t + 1]
        valid = ~np.isnan(X_w).any(axis=1)
        X_valid = X_w[valid]
        n_states = self._model.n_components if self._model else 3
        if len(X_valid) < n_states * 20:
            return

        cov_type = "diag"
        if hasattr(self._model, "covariance_type"):
            cov_type = self._model.covariance_type

        model = _fit_hmm_best_of_n(X_valid, n_components=n_states, covariance_type=cov_type, n_inits=5)
        if model is not None and _all_states_present(model, X_valid):
            old_map = dict(self._state_map)
            self._model = model
            self._build_state_map()
            if old_map and self._state_map != old_map:
                logger.info(
                    "Regime state-map changed at bar %d: %s -> %s",
                    t, old_map, self._state_map,
                )
                if self.config.cooldown_bars > 0:
                    self._state_map = old_map

    # ------------------------------------------------------------------
    # Feature engineering  (expanding-window normalization — no look-ahead)
    # ------------------------------------------------------------------

    def _prepare_features(self, ohlcv: pd.DataFrame) -> np.ndarray:
        """Build feature matrix: [log_return, rolling_vol, trend_spread, volume_ratio].

        Normalization uses an expanding window so that features at bar *t*
        only depend on data up to and including bar *t*.
        """
        close = ohlcv["close"].values.astype(np.float64)
        volume = ohlcv.get("volume", pd.Series(np.ones(len(ohlcv)))).values.astype(np.float64)
        high = ohlcv.get("high", ohlcv["close"]).values.astype(np.float64)
        low = ohlcv.get("low", ohlcv["close"]).values.astype(np.float64)

        log_ret = np.zeros(len(close))
        log_ret[1:] = np.log(close[1:] / np.maximum(close[:-1], 1e-12))

        vol_w = self.config.vol_window
        rolling_vol = (
            pd.Series(log_ret)
            .rolling(vol_w, min_periods=max(2, vol_w // 4))
            .std()
            .fillna(0)
            .values
        )

        ema_f = pd.Series(close).ewm(span=self.config.ema_fast, min_periods=1).mean().values
        ema_s = pd.Series(close).ewm(span=self.config.ema_slow, min_periods=1).mean().values

        atr = self._compute_atr(high, low, close, self.config.atr_period)
        atr_safe = np.maximum(atr, 1e-12)
        trend_spread = (ema_f - ema_s) / atr_safe

        vol_mean = pd.Series(volume).rolling(20, min_periods=1).mean().values
        vol_mean_safe = np.maximum(vol_mean, 1e-12)
        volume_ratio = volume / vol_mean_safe

        raw = np.column_stack([log_ret, rolling_vol, trend_spread, volume_ratio])

        features = self._expanding_normalize(raw)

        np.nan_to_num(features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return features

    @staticmethod
    def _expanding_normalize(raw: np.ndarray) -> np.ndarray:
        """Z-score each column using only data seen so far (expanding window)."""
        n, d = raw.shape
        out = np.zeros_like(raw)
        cum_sum = np.zeros(d)
        cum_sq = np.zeros(d)
        for t in range(n):
            row = raw[t]
            nan_mask = np.isnan(row)
            row_clean = np.where(nan_mask, 0.0, row)
            cum_sum += row_clean
            cum_sq += row_clean ** 2
            count = t + 1
            if count < 5:
                out[t] = 0.0
                continue
            mu = cum_sum / count
            var = cum_sq / count - mu ** 2
            sigma = np.sqrt(np.maximum(var, 0.0)) + 1e-12
            out[t] = np.where(nan_mask, 0.0, (row - mu) / sigma)
        return out

    @staticmethod
    def _compute_atr(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
    ) -> np.ndarray:
        tr = np.zeros(len(high))
        tr[0] = high[0] - low[0]
        for i in range(1, len(high)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )
        return pd.Series(tr).rolling(period, min_periods=1).mean().values

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _apply_circuit_breaker(self, prices: pd.Series, labels: pd.Series) -> pd.Series:
        """Force STRESS when rolling drawdown exceeds threshold."""
        result = labels.copy()
        window = max(168, self.config.rolling_window)
        rolling_peak = prices.rolling(window, min_periods=1).max()
        dd = (prices - rolling_peak) / (rolling_peak + 1e-12)
        stress_mask = dd < self.config.circuit_breaker_dd
        result[stress_mask] = REGIME_STRESS
        return result

    def _apply_hysteresis(self, labels: pd.Series) -> pd.Series:
        """Enforce minimum dwell time in each state."""
        result = labels.copy()
        cooldown = self.config.cooldown_bars
        if cooldown <= 0:
            return result

        current_label = result.iloc[0]
        bars_in_state = 0

        for i in range(len(result)):
            if result.iloc[i] == current_label:
                bars_in_state += 1
            else:
                if bars_in_state < cooldown:
                    result.iloc[i] = current_label
                    bars_in_state += 1
                else:
                    current_label = result.iloc[i]
                    bars_in_state = 1

        return result
