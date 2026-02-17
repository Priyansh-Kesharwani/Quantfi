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


def _simple_regime_detection(
    returns: np.ndarray,
    vol_threshold_percentile: float = 75.0
) -> np.ndarray:
    n = len(returns)
    window = min(21, n // 4)                   
    
    vol = np.full(n, np.nan)
    for i in range(window, n):
        vol[i] = np.std(returns[i - window:i])
    
    valid_vol = vol[~np.isnan(vol)]
    if len(valid_vol) == 0:
        return np.full(n, 0.5)
    
    threshold = np.percentile(valid_vol, vol_threshold_percentile)
    
    prob_stable = np.full(n, np.nan)
    for i in range(window, n):
        if not np.isnan(vol[i]):
            z = (vol[i] - threshold) / (threshold * 0.5 + 1e-10)
            prob_stable[i] = 1.0 / (1.0 + np.exp(z))
    
    return prob_stable


def _gmm_regime_detection(
    returns: np.ndarray,
    n_states: int = 2,
    window: int = 252,
    seed: int = 42
) -> np.ndarray:
    n = len(returns)
    prob_stable = np.full(n, np.nan)
    
    if n < window:
        return prob_stable
    
    np.random.seed(seed)
    
    full_returns = returns[~np.isnan(returns)]
    if len(full_returns) < window // 2:
        return prob_stable
    
    gmm_ref = GaussianMixture(
        n_components=n_states,
        covariance_type='full',
        random_state=seed,
        n_init=3,
        max_iter=100
    )
    
    try:
        gmm_ref.fit(full_returns.reshape(-1, 1))
    except Exception as e:
        logger.warning(f"GMM fitting failed: {e}")
        return _simple_regime_detection(returns)
    
    means = gmm_ref.means_.flatten()
    variances = gmm_ref.covariances_.flatten()
    
    mean_score = (means - means.min()) / (means.max() - means.min() + 1e-10)
    var_score = 1 - (variances - variances.min()) / (variances.max() - variances.min() + 1e-10)
    
    combined_score = mean_score * 0.6 + var_score * 0.4
    stable_state = np.argmax(combined_score)
    
    for i in range(window, n):
        window_returns = returns[i - window:i]
        valid_returns = window_returns[~np.isnan(window_returns)]
        
        if len(valid_returns) < window // 2:
            continue
        
        try:
            probs = gmm_ref.predict_proba(valid_returns[-1:].reshape(-1, 1))
            prob_stable[i] = probs[0, stable_state]
        except Exception:
            prob_stable[i] = gmm_ref.weights_[stable_state]
    
    return prob_stable


class HMMRegimeConfig:
    
    def __init__(
        self,
        n_states: int = 2,
        emission_type: str = "gaussian",
        window: int = 252,
        jump_penalty: Optional[float] = None,
        seed: int = 42
    ):
        self.n_states = n_states
        self.emission_type = emission_type
        self.window = window
        self.jump_penalty = jump_penalty
        self.seed = seed
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_states": self.n_states,
            "emission_type": self.emission_type,
            "window": self.window,
            "jump_penalty": self.jump_penalty,
            "seed": self.seed
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
            seed=config.seed
        )
        notes = "GMM-based regime detection"
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
