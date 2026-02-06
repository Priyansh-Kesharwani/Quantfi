"""
Phase 1 Development Endpoints

Read-only endpoints for inspecting Phase 1 indicators and composite calculations.
These endpoints use fixture data when no provider keys are present.

NON-NEGOTIABLE RULES:
- No prediction logic
- No live data unless explicitly enabled
- All indicator values include metadata for explainability
- Audit trail for every computation

Author: Phase 1 Implementation
Date: 2026-02-07
"""

import os
import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
import pandas as pd
import yaml
import logging

logger = logging.getLogger(__name__)

# Add project root to path for indicators import
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import Phase 1 indicators
try:
    from indicators.hurst import estimate_hurst
    from indicators.hmm_regime import infer_regime_prob, HMMRegimeConfig
    from indicators.vwap_z import compute_vwap_z
    from indicators.volatility import volatility_regime_score
    from indicators.liquidity import liquidity_score
    from indicators.coupling import coupling_score
    from indicators.normalization import normalize_to_score
    from indicators.composite import (
        compute_composite_score, Phase1Config, CompositeResult
    )
    INDICATORS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import indicators: {e}")
    INDICATORS_AVAILABLE = False

# Create router
phase1_router = APIRouter(prefix="/dev/phase1", tags=["Phase 1 Development"])


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class IndicatorValueResponse(BaseModel):
    """Response model for a single indicator value with metadata."""
    name: str
    raw_value: Optional[float] = None
    normalized_value: Optional[float] = None
    percentile: Optional[float] = None
    ecdf_sample_size: Optional[int] = None
    window_used: Optional[int] = None
    method: str
    notes: str


class Phase1IndicatorsResponse(BaseModel):
    """Response model for all Phase 1 indicators."""
    symbol: str
    timestamp: str
    data_source: str
    
    # Individual indicator values
    trend: Optional[IndicatorValueResponse] = None
    undervaluation: Optional[IndicatorValueResponse] = None
    volatility: Optional[IndicatorValueResponse] = None
    liquidity: Optional[IndicatorValueResponse] = None
    coupling: Optional[IndicatorValueResponse] = None
    hurst: Optional[IndicatorValueResponse] = None
    regime: Optional[IndicatorValueResponse] = None
    
    # Composite calculation steps
    composite: Optional[Dict[str, Any]] = None
    
    # Configuration used
    config: Dict[str, Any]


class Phase1HealthResponse(BaseModel):
    """Health check response for Phase 1 module."""
    status: str
    indicators_available: bool
    config_loaded: bool
    fixture_data_available: bool
    production_mode: bool
    timestamp: str


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_config() -> Phase1Config:
    """Load Phase 1 configuration from YAML."""
    config_path = PROJECT_ROOT / "config" / "phase1.yml"
    
    if config_path.exists():
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        return Phase1Config(
            regime_threshold=config_dict.get("regime_threshold"),
            regime_threshold_type=config_dict.get("regime_threshold_type", "hard"),
            S_scale=config_dict.get("S_scale"),
            committee_method=config_dict.get("committee_method", "mean"),
            committee_trim_pct=config_dict.get("committee_trim_pct", 0.1),
            g_pers_type=config_dict.get("g_pers_type", "sigmoid_canonical"),
            g_pers_params=config_dict.get("g_pers_params", {"k": 10.0, "H_neutral": 0.5}),
            min_obs=config_dict.get("normalization", {}).get("min_obs", 100),
            log_intermediates=config_dict.get("logging", {}).get("log_intermediates", True),
            log_path=config_dict.get("logging", {}).get("log_path", "logs/phase1_indicator_runs.json"),
            allow_production_mode=config_dict.get("allow_production_mode", False)
        )
    else:
        logger.warning("Config file not found, using defaults")
        return Phase1Config()


def load_fixture_data(symbol: str) -> Optional[pd.DataFrame]:
    """
    Load fixture data for an asset.
    
    Looks for CSV files in tests/fixtures/ directory.
    """
    fixture_dir = PROJECT_ROOT / "tests" / "fixtures"
    
    # Try different file patterns
    patterns = [
        f"{symbol.lower()}_prices.csv",
        f"{symbol.upper()}_prices.csv",
        f"{symbol}_prices.csv",
        "synthetic_prices.csv",  # Fallback to generic synthetic data
    ]
    
    for pattern in patterns:
        filepath = fixture_dir / pattern
        if filepath.exists():
            try:
                df = pd.read_csv(filepath, parse_dates=["date"], index_col="date")
                return df
            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")
    
    return None


def generate_synthetic_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic price/volume data for testing.
    
    This is used when no fixture data is available.
    """
    np.random.seed(seed)
    
    # Generate dates
    dates = pd.date_range(end=datetime.now(), periods=n, freq="D")
    
    # Generate price series (geometric Brownian motion with drift)
    returns = np.random.randn(n) * 0.02 + 0.0003  # ~7% annual return
    price = 100 * np.exp(np.cumsum(returns))
    
    # Generate volume (correlated with absolute returns)
    base_volume = 1_000_000
    volume = base_volume * (1 + np.abs(returns) * 10 + np.random.randn(n) * 0.3)
    volume = np.maximum(volume, 100_000)
    
    df = pd.DataFrame({
        "close": price,
        "volume": volume,
        "high": price * (1 + np.abs(np.random.randn(n)) * 0.01),
        "low": price * (1 - np.abs(np.random.randn(n)) * 0.01),
        "open": np.roll(price, 1)
    }, index=dates)
    df.columns = ["close", "volume", "high", "low", "open"]
    
    return df


# =============================================================================
# ENDPOINTS
# =============================================================================

@phase1_router.get("/health", response_model=Phase1HealthResponse)
async def phase1_health():
    """
    Health check for Phase 1 module.
    
    Returns status of:
    - Indicator module availability
    - Configuration loading
    - Fixture data availability
    - Production mode flag
    """
    config = load_config()
    fixture_path = PROJECT_ROOT / "tests" / "fixtures"
    
    return Phase1HealthResponse(
        status="healthy" if INDICATORS_AVAILABLE else "degraded",
        indicators_available=INDICATORS_AVAILABLE,
        config_loaded=True,  # load_config always returns something
        fixture_data_available=fixture_path.exists(),
        production_mode=config.allow_production_mode,
        timestamp=datetime.utcnow().isoformat()
    )


@phase1_router.get("/indicators/{asset}", response_model=Phase1IndicatorsResponse)
async def get_phase1_indicators(asset: str):
    """
    Get Phase 1 indicators and composite score for an asset.
    
    This endpoint uses fixture data when no provider keys are present.
    It computes all Phase 1 indicators and the symbolic composite.
    
    Parameters
    ----------
    asset : str
        Asset symbol (e.g., "GOLD", "BTC", "SPY")
    
    Returns
    -------
    Phase1IndicatorsResponse
        All indicator values with metadata and composite calculation steps
    
    Notes
    -----
    - Uses fixture/synthetic data in Phase 1 (no live data)
    - All numeric weights/thresholds are symbolic (not tuned)
    - Returns full metadata for explainability
    """
    if not INDICATORS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Phase 1 indicators module not available"
        )
    
    # Load configuration
    config = load_config()
    
    # Check production mode
    if config.allow_production_mode:
        raise HTTPException(
            status_code=403,
            detail="Production mode is enabled but Phase 1 should use fixtures only"
        )
    
    # Load data
    df = load_fixture_data(asset)
    data_source = f"fixture:{asset.lower()}_prices.csv"
    
    if df is None:
        # Generate synthetic data
        df = generate_synthetic_data(n=500, seed=42)
        data_source = "synthetic:generated"
    
    # Extract series
    prices = df["close"].values
    volumes = df.get("volume", pd.Series(dtype=float)).values
    n = len(prices)
    
    # Compute all indicators
    indicator_results = {}
    
    # 1. Hurst exponent
    try:
        H_t, H_meta = estimate_hurst(prices, window=252)
        latest_H = H_t[-1] if not np.isnan(H_t[-1]) else None
        
        # Normalize
        if latest_H is not None:
            H_norm_series, H_norm_meta = normalize_to_score(
                H_t, min_obs=config.min_obs, higher_is_favorable=True
            )
            H_normalized = H_norm_series[-1] if not np.isnan(H_norm_series[-1]) else None
        else:
            H_normalized = None
        
        indicator_results["hurst"] = IndicatorValueResponse(
            name="Hurst Exponent",
            raw_value=latest_H,
            normalized_value=H_normalized,
            percentile=None,
            ecdf_sample_size=n,
            window_used=H_meta["window_used"],
            method=H_meta["method"],
            notes=H_meta["notes"]
        )
    except Exception as e:
        logger.warning(f"Hurst calculation failed: {e}")
        indicator_results["hurst"] = IndicatorValueResponse(
            name="Hurst Exponent",
            method="error",
            notes=f"Calculation failed: {str(e)}"
        )
    
    # 2. HMM Regime
    try:
        hmm_config = HMMRegimeConfig(n_states=2, window=252, seed=42)
        R_t, R_meta = infer_regime_prob(prices, hmm_config)
        latest_R = R_t[-1] if not np.isnan(R_t[-1]) else None
        
        indicator_results["regime"] = IndicatorValueResponse(
            name="Regime Probability",
            raw_value=latest_R,
            normalized_value=latest_R,  # Already in [0, 1]
            percentile=None,
            ecdf_sample_size=n,
            window_used=R_meta["window_used"],
            method=R_meta["method"],
            notes=R_meta["notes"]
        )
    except Exception as e:
        logger.warning(f"Regime calculation failed: {e}")
        indicator_results["regime"] = IndicatorValueResponse(
            name="Regime Probability",
            method="error",
            notes=f"Calculation failed: {str(e)}"
        )
    
    # 3. VWAP Z-score (Undervaluation)
    try:
        Z_t, Z_meta = compute_vwap_z(prices, volumes if len(volumes) == n else None, window=20)
        latest_Z = Z_t[-1] if not np.isnan(Z_t[-1]) else None
        
        # Normalize (invert: negative Z = undervalued = favorable)
        if latest_Z is not None:
            U_t = -Z_t  # Invert
            U_norm_series, U_norm_meta = normalize_to_score(
                U_t, min_obs=config.min_obs, higher_is_favorable=True
            )
            U_normalized = U_norm_series[-1] if not np.isnan(U_norm_series[-1]) else None
        else:
            U_normalized = None
        
        indicator_results["undervaluation"] = IndicatorValueResponse(
            name="VWAP Z-Score",
            raw_value=latest_Z,
            normalized_value=U_normalized,
            percentile=None,
            ecdf_sample_size=n,
            window_used=Z_meta["window_used"],
            method=Z_meta["method"],
            notes=Z_meta["notes"]
        )
    except Exception as e:
        logger.warning(f"VWAP Z calculation failed: {e}")
        indicator_results["undervaluation"] = IndicatorValueResponse(
            name="VWAP Z-Score",
            method="error",
            notes=f"Calculation failed: {str(e)}"
        )
    
    # 4. Volatility Regime
    try:
        V_t, V_meta = volatility_regime_score(prices, vol_window=21, pct_lookback=252)
        latest_V = V_t[-1] if not np.isnan(V_t[-1]) else None
        
        indicator_results["volatility"] = IndicatorValueResponse(
            name="Volatility Regime",
            raw_value=latest_V,
            normalized_value=latest_V,  # Already in [0, 1]
            percentile=None,
            ecdf_sample_size=n,
            window_used=V_meta["window_used"],
            method=V_meta["method"],
            notes=V_meta["notes"]
        )
    except Exception as e:
        logger.warning(f"Volatility calculation failed: {e}")
        indicator_results["volatility"] = IndicatorValueResponse(
            name="Volatility Regime",
            method="error",
            notes=f"Calculation failed: {str(e)}"
        )
    
    # 5. Liquidity
    try:
        L_t, L_meta = liquidity_score(prices, volumes if len(volumes) == n else None, window=21)
        latest_L = L_t[-1] if not np.isnan(L_t[-1]) else None
        
        indicator_results["liquidity"] = IndicatorValueResponse(
            name="Liquidity Score",
            raw_value=latest_L,
            normalized_value=latest_L,  # Already in [0, 1]
            percentile=None,
            ecdf_sample_size=n,
            window_used=L_meta["window_used"],
            method=L_meta["method"],
            notes=L_meta["notes"]
        )
    except Exception as e:
        logger.warning(f"Liquidity calculation failed: {e}")
        indicator_results["liquidity"] = IndicatorValueResponse(
            name="Liquidity Score",
            method="error",
            notes=f"Calculation failed: {str(e)}"
        )
    
    # 6. Coupling (no market data in fixture, returns neutral)
    try:
        # Compute returns
        returns = np.diff(np.log(np.maximum(prices, 1e-10)))
        returns = np.insert(returns, 0, 0)
        
        C_t, C_meta = coupling_score(returns, market_returns=None, window=63)
        latest_C = C_t[-1] if not np.isnan(C_t[-1]) else None
        
        indicator_results["coupling"] = IndicatorValueResponse(
            name="Systemic Coupling",
            raw_value=latest_C,
            normalized_value=latest_C,  # Already in [0, 1]
            percentile=None,
            ecdf_sample_size=n,
            window_used=C_meta["window_used"],
            method=C_meta["method"],
            notes=C_meta["notes"]
        )
    except Exception as e:
        logger.warning(f"Coupling calculation failed: {e}")
        indicator_results["coupling"] = IndicatorValueResponse(
            name="Systemic Coupling",
            method="error",
            notes=f"Calculation failed: {str(e)}"
        )
    
    # 7. Trend (simplified - use SMA crossover as proxy)
    try:
        # Simple trend: price vs 50-day SMA
        sma_50 = pd.Series(prices).rolling(50).mean().values
        trend_raw = (prices - sma_50) / sma_50  # Deviation from SMA
        
        T_norm_series, T_norm_meta = normalize_to_score(
            trend_raw, min_obs=config.min_obs, higher_is_favorable=True
        )
        latest_T = T_norm_series[-1] if not np.isnan(T_norm_series[-1]) else None
        
        indicator_results["trend"] = IndicatorValueResponse(
            name="Trend Score",
            raw_value=trend_raw[-1] if not np.isnan(trend_raw[-1]) else None,
            normalized_value=latest_T,
            percentile=None,
            ecdf_sample_size=n,
            window_used=50,
            method="sma_deviation",
            notes="Trend via price deviation from 50-day SMA (normalized)"
        )
    except Exception as e:
        logger.warning(f"Trend calculation failed: {e}")
        indicator_results["trend"] = IndicatorValueResponse(
            name="Trend Score",
            method="error",
            notes=f"Calculation failed: {str(e)}"
        )
    
    # Compute composite
    composite_result = None
    try:
        # Get normalized values (default to 0.5 if missing)
        T_t = indicator_results.get("trend", {})
        T_val = T_t.normalized_value if hasattr(T_t, 'normalized_value') and T_t.normalized_value is not None else 0.5
        
        U_t = indicator_results.get("undervaluation", {})
        U_val = U_t.normalized_value if hasattr(U_t, 'normalized_value') and U_t.normalized_value is not None else 0.5
        
        V_t = indicator_results.get("volatility", {})
        V_val = V_t.normalized_value if hasattr(V_t, 'normalized_value') and V_t.normalized_value is not None else 0.5
        
        L_t = indicator_results.get("liquidity", {})
        L_val = L_t.normalized_value if hasattr(L_t, 'normalized_value') and L_t.normalized_value is not None else 0.5
        
        C_t = indicator_results.get("coupling", {})
        C_val = C_t.normalized_value if hasattr(C_t, 'normalized_value') and C_t.normalized_value is not None else 0.5
        
        H_t = indicator_results.get("hurst", {})
        H_val = H_t.normalized_value if hasattr(H_t, 'normalized_value') and H_t.normalized_value is not None else 0.5
        
        R_t = indicator_results.get("regime", {})
        R_val = R_t.normalized_value if hasattr(R_t, 'normalized_value') and R_t.normalized_value is not None else 0.5
        
        result: CompositeResult = compute_composite_score(
            T_t=T_val, U_t=U_val, V_t=V_val, L_t=L_val, C_t=C_val,
            H_t=H_val, R_t=R_val, config=config
        )
        
        composite_result = {
            "score": result.score,
            "interpretation": _interpret_score(result.score),
            "intermediates": {
                "g_pers_H": result.g_pers_H,
                "Gate_t": result.Gate_t,
                "Opp_t": result.Opp_t,
                "RawFavor_t": result.RawFavor_t
            },
            "formula": "score = 100 * clip(0.5 + (RawFavor - 0.5) * S_scale, 0, 1)",
            "notes": "Phase 1 symbolic composite - weights/thresholds not tuned"
        }
    except Exception as e:
        logger.warning(f"Composite calculation failed: {e}")
        composite_result = {
            "score": 50.0,
            "interpretation": "neutral (calculation error)",
            "error": str(e)
        }
    
    return Phase1IndicatorsResponse(
        symbol=asset.upper(),
        timestamp=datetime.utcnow().isoformat(),
        data_source=data_source,
        trend=indicator_results.get("trend"),
        undervaluation=indicator_results.get("undervaluation"),
        volatility=indicator_results.get("volatility"),
        liquidity=indicator_results.get("liquidity"),
        coupling=indicator_results.get("coupling"),
        hurst=indicator_results.get("hurst"),
        regime=indicator_results.get("regime"),
        composite=composite_result,
        config=config.to_dict()
    )


def _interpret_score(score: float) -> str:
    """Interpret composite score."""
    if score >= 70:
        return "favorable - consider increasing DCA intensity"
    elif score >= 55:
        return "slightly favorable"
    elif score >= 45:
        return "neutral - maintain baseline DCA"
    elif score >= 30:
        return "slightly unfavorable"
    else:
        return "unfavorable - maintain minimal DCA (baseline never stops)"


@phase1_router.get("/config")
async def get_phase1_config():
    """
    Get current Phase 1 configuration.
    
    Shows all configuration parameters including symbolic placeholders.
    """
    config_path = PROJECT_ROOT / "config" / "phase1.yml"
    
    if config_path.exists():
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return {
            "source": str(config_path),
            "config": config_dict
        }
    else:
        return {
            "source": "defaults",
            "config": Phase1Config().to_dict()
        }
