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
from app_config import get_backend_config

logger = logging.getLogger(__name__)
CFG = get_backend_config()

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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

phase1_router = APIRouter(prefix="/dev/phase1", tags=["Phase 1 Development"])


class IndicatorValueResponse(BaseModel):
    name: str
    raw_value: Optional[float] = None
    normalized_value: Optional[float] = None
    percentile: Optional[float] = None
    ecdf_sample_size: Optional[int] = None
    window_used: Optional[int] = None
    method: str
    notes: str


class Phase1IndicatorsResponse(BaseModel):
    symbol: str
    timestamp: str
    data_source: str
    
    trend: Optional[IndicatorValueResponse] = None
    undervaluation: Optional[IndicatorValueResponse] = None
    volatility: Optional[IndicatorValueResponse] = None
    liquidity: Optional[IndicatorValueResponse] = None
    coupling: Optional[IndicatorValueResponse] = None
    hurst: Optional[IndicatorValueResponse] = None
    regime: Optional[IndicatorValueResponse] = None
    
    composite: Optional[Dict[str, Any]] = None
    
    config: Dict[str, Any]


class Phase1HealthResponse(BaseModel):
    status: str
    indicators_available: bool
    config_loaded: bool
    fixture_data_available: bool
    production_mode: bool
    timestamp: str


def load_config() -> Phase1Config:
    config_dict = _load_phase1_raw_config()
    if not config_dict:
        logger.warning("Config file not found, using defaults")
        return Phase1Config()
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


def _load_phase1_raw_config() -> Dict[str, Any]:
    config_path = PROJECT_ROOT / "config" / "phase1.yml"
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def load_fixture_data(symbol: str) -> Optional[pd.DataFrame]:
    fixture_dir = PROJECT_ROOT / "tests" / "fixtures"
    
    patterns = [
        f"{symbol.lower()}_prices.csv",
        f"{symbol.upper()}_prices.csv",
        f"{symbol}_prices.csv",
        "synthetic_prices.csv",                                      
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


@phase1_router.get("/health", response_model=Phase1HealthResponse)
async def phase1_health():
    config = load_config()
    fixture_path = PROJECT_ROOT / "tests" / "fixtures"
    
    return Phase1HealthResponse(
        status="healthy" if INDICATORS_AVAILABLE else "degraded",
        indicators_available=INDICATORS_AVAILABLE,
        config_loaded=True,                                        
        fixture_data_available=fixture_path.exists(),
        production_mode=config.allow_production_mode,
        timestamp=datetime.utcnow().isoformat()
    )


@phase1_router.get("/indicators/{asset}", response_model=Phase1IndicatorsResponse)
async def get_phase1_indicators(asset: str):
    if not INDICATORS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Indicators module not available"
        )
    
    config = load_config()
    config_dict = _load_phase1_raw_config()
    
    if config.allow_production_mode:
        raise HTTPException(
            status_code=403,
            detail="Production mode is enabled but dev routes should use fixtures only"
        )
    
    df = load_fixture_data(asset)
    data_source = f"fixture:{asset.lower()}_prices.csv"
    
    if df is None:
        try:
            from data_providers import PriceProvider
            yf_df = PriceProvider.fetch_historical_data(asset, period=CFG.history_period)
            if yf_df is not None and not yf_df.empty:
                yf_df.columns = [c.lower() for c in yf_df.columns]
                df = yf_df
                data_source = f"yfinance:{asset}"
        except Exception as e:
            logger.warning(f"yfinance fallback failed for indicators: {e}")

    if df is None:
        raise HTTPException(
            status_code=404,
            detail=f"No real data available for {asset}. Add it to the watchlist first."
        )
    
    prices = df["close"].values
    volumes = df.get("volume", pd.Series(dtype=float)).values
    n = len(prices)
    windows_cfg = {}
    windows_cfg = config_dict.get("windows", {})
    norm_cfg = config_dict.get("normalization", {})
    hmm_cfg = config_dict.get("hmm", {})
    hurst_w = int(windows_cfg.get("hurst", 252))
    hmm_w = int(windows_cfg.get("hmm_regime", 252))
    vwap_w = int(windows_cfg.get("vwap", 20))
    vol_w = int(windows_cfg.get("volatility", 21))
    vol_pct_w = int(windows_cfg.get("vol_percentile", 252))
    liq_w = int(windows_cfg.get("liquidity", 21))
    coupling_w = int(windows_cfg.get("coupling", 63))
    min_obs = int(norm_cfg.get("min_obs", config.min_obs))
    hmm_states = int(hmm_cfg.get("n_states", 2))
    hmm_seed = int(hmm_cfg.get("seed", 42))
    
    indicator_results = {}
    
    try:
        H_t, H_meta = estimate_hurst(prices, window=hurst_w)
        latest_H = H_t[-1] if not np.isnan(H_t[-1]) else None
        
        if latest_H is not None:
            H_norm_series, H_norm_meta = normalize_to_score(
                H_t, min_obs=min_obs, higher_is_favorable=True
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
    
    try:
        hmm_config = HMMRegimeConfig(n_states=hmm_states, window=hmm_w, seed=hmm_seed)
        R_t, R_meta = infer_regime_prob(prices, hmm_config)
        latest_R = R_t[-1] if not np.isnan(R_t[-1]) else None
        
        indicator_results["regime"] = IndicatorValueResponse(
            name="Regime Probability",
            raw_value=latest_R,
            normalized_value=latest_R,                     
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
    
    try:
        Z_t, Z_meta = compute_vwap_z(prices, volumes if len(volumes) == n else None, window=vwap_w)
        latest_Z = Z_t[-1] if not np.isnan(Z_t[-1]) else None
        
        if latest_Z is not None:
            U_t = -Z_t          
            U_norm_series, U_norm_meta = normalize_to_score(
                U_t, min_obs=min_obs, higher_is_favorable=True
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
    
    try:
        V_t, V_meta = volatility_regime_score(prices, vol_window=vol_w, pct_lookback=vol_pct_w)
        latest_V = V_t[-1] if not np.isnan(V_t[-1]) else None
        
        indicator_results["volatility"] = IndicatorValueResponse(
            name="Volatility Regime",
            raw_value=latest_V,
            normalized_value=latest_V,                     
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
    
    try:
        L_t, L_meta = liquidity_score(prices, volumes if len(volumes) == n else None, window=liq_w)
        latest_L = L_t[-1] if not np.isnan(L_t[-1]) else None
        
        indicator_results["liquidity"] = IndicatorValueResponse(
            name="Liquidity Score",
            raw_value=latest_L,
            normalized_value=latest_L,                     
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
    
    try:
        returns = np.diff(np.log(np.maximum(prices, 1e-10)))
        returns = np.insert(returns, 0, 0)
        
        C_t, C_meta = coupling_score(returns, market_returns=None, window=coupling_w)
        latest_C = C_t[-1] if not np.isnan(C_t[-1]) else None
        
        indicator_results["coupling"] = IndicatorValueResponse(
            name="Systemic Coupling",
            raw_value=latest_C,
            normalized_value=latest_C,                     
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
    
    try:
        trend_window = int(CFG.phase1_trend_sma_window)
        sma_series = pd.Series(prices).rolling(trend_window).mean().values
        trend_raw = (prices - sma_series) / sma_series
        
        T_norm_series, T_norm_meta = normalize_to_score(
            trend_raw, min_obs=min_obs, higher_is_favorable=True
        )
        latest_T = T_norm_series[-1] if not np.isnan(T_norm_series[-1]) else None
        
        indicator_results["trend"] = IndicatorValueResponse(
            name="Trend Score",
            raw_value=trend_raw[-1] if not np.isnan(trend_raw[-1]) else None,
            normalized_value=latest_T,
            percentile=None,
            ecdf_sample_size=n,
            window_used=trend_window,
            method="sma_deviation",
            notes=f"Trend via price deviation from {trend_window}-day SMA (normalized)"
        )
    except Exception as e:
        logger.warning(f"Trend calculation failed: {e}")
        indicator_results["trend"] = IndicatorValueResponse(
            name="Trend Score",
            method="error",
            notes=f"Calculation failed: {str(e)}"
        )
    
    composite_result = None
    try:
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
            "notes": "Symbolic composite — weights/thresholds configurable"
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
    bands = CFG.phase1_score_bands
    if score >= bands['favorable']:
        return "favorable - consider increasing DCA intensity"
    elif score >= bands['slightly_favorable']:
        return "slightly favorable"
    elif score >= bands['neutral']:
        return "neutral - maintain baseline DCA"
    elif score >= bands['slightly_unfavorable']:
        return "slightly unfavorable"
    else:
        return "unfavorable - maintain minimal DCA (baseline never stops)"


@phase1_router.get("/config")
async def get_phase1_config():
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
