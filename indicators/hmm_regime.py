"""
HMM Regime Detection

Hidden Markov Model for detecting market regime states. The primary output
is P(state=StableExpansion), which represents the probability that the market
is in a stable expansion regime (favorable for DCA).

Phase 1 Implementation Notes:
-----------------------------
This is a STUB implementation using Gaussian Mixture Model (GMM) to approximate
regime detection. In Phase 2, this should be replaced with:
- Full HMM with t-distributed emissions (heavier tails)
- Jump penalty for regime switching
- Proper Viterbi path for state sequence

References:
-----------
1. Hamilton, J.D. (1989). "A new approach to the economic analysis of 
   nonstationary time series and the business cycle"
2. Ang, A. & Bekaert, G. (2002). "Regime switches in interest rates"
3. Guidolin, M. & Timmermann, A. (2007). "Asset allocation under multivariate
   regime switching"

Author: Phase 1 Implementation
Date: 2026-02-07
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import sklearn for GMM
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
    """
    Simple threshold-based regime detection fallback.
    
    High volatility regime when rolling volatility exceeds threshold.
    
    Parameters
    ----------
    returns : np.ndarray
        Log returns series
    vol_threshold_percentile : float
        Percentile threshold for high volatility regime
    
    Returns
    -------
    np.ndarray
        P(StableExpansion) - probability of stable regime
    """
    n = len(returns)
    window = min(21, n // 4)  # ~1 month window
    
    # Rolling volatility
    vol = np.full(n, np.nan)
    for i in range(window, n):
        vol[i] = np.std(returns[i - window:i])
    
    # Threshold
    valid_vol = vol[~np.isnan(vol)]
    if len(valid_vol) == 0:
        return np.full(n, 0.5)
    
    threshold = np.percentile(valid_vol, vol_threshold_percentile)
    
    # P(Stable) = 1 - sigmoid(vol - threshold)
    # Higher vol = lower probability of stable expansion
    prob_stable = np.full(n, np.nan)
    for i in range(window, n):
        if not np.isnan(vol[i]):
            # Sigmoid transformation
            z = (vol[i] - threshold) / (threshold * 0.5 + 1e-10)
            prob_stable[i] = 1.0 / (1.0 + np.exp(z))
    
    return prob_stable


def _gmm_regime_detection(
    returns: np.ndarray,
    n_states: int = 2,
    window: int = 252,
    seed: int = 42
) -> np.ndarray:
    """
    GMM-based regime detection (Phase 1 stub for HMM).
    
    Fits a Gaussian Mixture Model to rolling windows of returns
    and identifies the "stable expansion" state as the one with
    positive mean and moderate volatility.
    
    Parameters
    ----------
    returns : np.ndarray
        Log returns series
    n_states : int
        Number of regime states (default: 2)
    window : int
        Rolling window for local GMM fit
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    np.ndarray
        P(StableExpansion) for each time point
    
    TODO (Phase 2):
    - Replace GMM with full HMM implementation
    - Use t-distributed emissions for heavier tails
    - Add jump penalty for regime transitions
    - Implement proper Viterbi algorithm for state sequence
    """
    n = len(returns)
    prob_stable = np.full(n, np.nan)
    
    # Need enough history
    if n < window:
        return prob_stable
    
    # Fit GMM on full history first to identify state labels
    # This gives us a reference for which state is "stable expansion"
    np.random.seed(seed)
    
    full_returns = returns[~np.isnan(returns)]
    if len(full_returns) < window // 2:
        return prob_stable
    
    # Fit reference GMM
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
    
    # Identify stable expansion state:
    # - Higher mean (positive returns)
    # - Lower variance (stability)
    means = gmm_ref.means_.flatten()
    variances = gmm_ref.covariances_.flatten()
    
    # Score: prefer high mean, low variance
    # Normalize to [0,1] and combine
    mean_score = (means - means.min()) / (means.max() - means.min() + 1e-10)
    var_score = 1 - (variances - variances.min()) / (variances.max() - variances.min() + 1e-10)
    
    combined_score = mean_score * 0.6 + var_score * 0.4
    stable_state = np.argmax(combined_score)
    
    # Rolling probability estimation
    for i in range(window, n):
        window_returns = returns[i - window:i]
        valid_returns = window_returns[~np.isnan(window_returns)]
        
        if len(valid_returns) < window // 2:
            continue
        
        try:
            # Predict probabilities for latest point
            probs = gmm_ref.predict_proba(valid_returns[-1:].reshape(-1, 1))
            prob_stable[i] = probs[0, stable_state]
        except Exception:
            # Fallback: use prior probabilities
            prob_stable[i] = gmm_ref.weights_[stable_state]
    
    return prob_stable


class HMMRegimeConfig:
    """Configuration for HMM regime detection."""
    
    def __init__(
        self,
        n_states: int = 2,
        emission_type: str = "gaussian",  # TODO Phase 2: "student_t"
        window: int = 252,
        jump_penalty: Optional[float] = None,  # TODO Phase 2
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
    """
    Compute P(state=StableExpansion) using HMM regime detection.
    
    Parameters
    ----------
    series : np.ndarray
        1D array of prices (not returns). The function will compute
        log returns internally.
    config : HMMRegimeConfig, optional
        Configuration object. If None, uses defaults.
    
    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        - R_t: Array of regime probabilities (same length as input)
        - meta: Metadata dict with keys:
            - window_used: int
            - n_obs: int
            - method: str
            - seed: int
            - notes: str
    
    Examples
    --------
    >>> import numpy as np
    >>> prices = np.cumprod(1 + np.random.randn(500) * 0.01 + 0.0002)
    >>> config = HMMRegimeConfig(n_states=2, seed=42)
    >>> R_t, meta = infer_regime_prob(prices, config)
    >>> print(f"Latest P(Stable): {R_t[-1]:.3f}")
    
    Notes
    -----
    Phase 1 uses a GMM stub. Phase 2 should implement:
    - Full HMM with forward-backward algorithm
    - t-distributed emissions for heavy tails
    - Jump penalty to discourage frequent regime switches
    """
    if config is None:
        config = HMMRegimeConfig()
    
    series = np.asarray(series, dtype=np.float64)
    n = len(series)
    
    # Compute log returns
    safe_prices = np.maximum(series, 1e-10)
    log_returns = np.diff(np.log(safe_prices))
    
    # Prepend NaN to align with original series length
    log_returns = np.insert(log_returns, 0, np.nan)
    
    # Choose method
    notes = ""
    if SKLEARN_AVAILABLE:
        method = "gmm_stub"
        R_t = _gmm_regime_detection(
            log_returns,
            n_states=config.n_states,
            window=config.window,
            seed=config.seed
        )
        notes = "Phase 1 GMM stub. TODO: Replace with full HMM + t-emissions in Phase 2"
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


# Convenience function
def regime_probability(series: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Alias for infer_regime_prob with keyword config."""
    config = HMMRegimeConfig(**kwargs)
    return infer_regime_prob(series, config)
