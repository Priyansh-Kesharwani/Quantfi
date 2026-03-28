import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

try:
    from sklearn.mixture import GaussianMixture
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available. Using simple threshold-based regime detection.")


DEFAULT_VOL_WINDOW = 21
FALLBACK_REGIME = 0.5


def _simple_regime_detection(
    returns: np.ndarray,
    vol_threshold_percentile: float = 75.0
) -> np.ndarray:
    n = len(returns)
    window = min(DEFAULT_VOL_WINDOW, n // 4) if n else DEFAULT_VOL_WINDOW
    vol = np.full(n, np.nan)
    for i in range(window, n):
        vol[i] = np.std(returns[i - window:i])
    prob_stable = np.full(n, FALLBACK_REGIME)
    for i in range(window, n):
        if np.isnan(vol[i]):
            continue
        hist_vol = vol[window:i]
        valid_hist = hist_vol[~np.isnan(hist_vol)]
        if len(valid_hist) < window // 2:
            continue
        threshold_i = np.percentile(valid_hist, vol_threshold_percentile)
        z = (vol[i] - threshold_i) / (threshold_i * 0.5 + 1e-10)
        prob_stable[i] = 1.0 / (1.0 + np.exp(z))
    return prob_stable


MIN_SAMPLES_DEFAULT = 2


def _gmm_regime_detection(
    returns: np.ndarray,
    n_states: int = 2,
    window: int = 252,
    seed: int = 42,
    min_samples: Optional[int] = None,
) -> np.ndarray:
    n = len(returns)
    prob_stable = np.full(n, np.nan)
    min_samp = min_samples if min_samples is not None else max(window // 2, n_states * MIN_SAMPLES_DEFAULT)
    if n < window:
        return prob_stable
    for i in range(window, n):
        past = returns[i - window:i]
        valid = past[~np.isnan(past)]
        if len(valid) < min_samp:
            prob_stable[i] = FALLBACK_REGIME
            continue
        X = valid.reshape(-1, 1)
        rng = np.random.RandomState(seed)
        gmm = GaussianMixture(
            n_components=n_states,
            covariance_type="full",
            random_state=rng,
            n_init=3,
            max_iter=100,
        )
        try:
            gmm.fit(X)
        except Exception as e:
            logger.debug("GMM fit failed at i=%s: %s", i, e)
            prob_stable[i] = FALLBACK_REGIME
            continue
        means = gmm.means_.flatten()
        variances = gmm.covariances_.flatten()
        mean_score = (means - means.min()) / (means.max() - means.min() + 1e-10)
        var_score = 1.0 - (variances - variances.min()) / (variances.max() - variances.min() + 1e-10)
        combined = mean_score * 0.6 + var_score * 0.4
        stable_state = np.argmax(combined)
        try:
            probs = gmm.predict_proba(valid[-1:].reshape(-1, 1))
            prob_stable[i] = float(probs[0, stable_state])
        except Exception:
            prob_stable[i] = float(gmm.weights_[stable_state])
    return prob_stable


class HMMRegimeConfig:
    def __init__(
        self,
        n_states: int = 2,
        emission_type: str = "gaussian",
        window: int = 252,
        jump_penalty: Optional[float] = None,
        seed: int = 42,
        min_samples: Optional[int] = None,
    ):
        self.n_states = n_states
        self.emission_type = emission_type
        self.window = window
        self.jump_penalty = jump_penalty
        self.seed = seed
        self.min_samples = min_samples

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_states": self.n_states,
            "emission_type": self.emission_type,
            "window": self.window,
            "jump_penalty": self.jump_penalty,
            "seed": self.seed,
            "min_samples": self.min_samples,
        }


def infer_regime_prob(
    series: np.ndarray,
    config: Optional[HMMRegimeConfig] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if config is None:
        config = HMMRegimeConfig()
    
    series = np.asarray(series, dtype=np.float64)
    n = len(series)
    
    safe_prices = np.maximum(series, 1e-10)
    log_returns = np.diff(np.log(safe_prices))
    
    log_returns = np.insert(log_returns, 0, np.nan)
    
    notes = ""
    if SKLEARN_AVAILABLE:
        method = "gmm_stub"
        R_t = _gmm_regime_detection(
            log_returns,
            n_states=config.n_states,
            window=config.window,
            seed=config.seed,
            min_samples=config.min_samples,
        )
        notes = "GMM-based regime detection (rolling fit, past-only)"
    else:
        method = "threshold_fallback"
        R_t = _simple_regime_detection(log_returns)
        notes = "sklearn not available. Using simple threshold-based detection."
    
    meta = {
        "window_used": config.window,
        "n_obs": n,
        "method": method,
        "seed": config.seed,
        "notes": notes,
        "config": config.to_dict()
    }
    
    return R_t, meta


def regime_probability(series: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
    config = HMMRegimeConfig(**kwargs)
    return infer_regime_prob(series, config)
