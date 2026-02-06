"""
Unified Indicator Engine

Computes all indicator components from raw OHLCV data and returns
normalized scores ready for composite calculation.

Components:
- T_t: Trend strength [0, 1]
- U_t: Undervaluation [0, 1]
- H_t: Hurst exponent (persistence) [0, 1]
- V_t: Inverse volatility (calm market) [0, 1]
- L_t: Liquidity [0, 1]
- C_t: Systemic decoupling [0, 1]
- R_t: Regime stability [0, 1]

Author: Phase 2 Implementation
Date: 2026-02-07
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# Import component indicators
from .trend import trend_strength_score
from .undervaluation import undervaluation_score
from .hurst import estimate_hurst
from .volatility import volatility_regime_score
from .liquidity import liquidity_score
from .coupling import coupling_score
from .hmm_regime import infer_regime_prob, HMMRegimeConfig


@dataclass
class IndicatorConfig:
    """Configuration for all indicators."""
    
    # Trend
    trend_ema_short: int = 12
    trend_ema_long: int = 26
    trend_adx_window: int = 14
    trend_method: str = "combined"
    
    # Undervaluation
    uval_vwap_window: int = 20
    uval_zscore_window: int = 50
    uval_dd_lookback: int = 252
    uval_method: str = "combined"
    
    # Hurst
    hurst_window: int = 252
    hurst_method: str = "auto"
    
    # Volatility
    vol_window: int = 21
    vol_pct_lookback: int = 252
    
    # Liquidity
    liq_window: int = 21
    liq_pct_lookback: int = 252
    
    # Coupling
    coupling_window: int = 63
    coupling_method: str = "auto"
    
    # Regime
    regime_n_states: int = 2
    regime_window: int = 252
    regime_seed: int = 42
    
    # Normalization
    min_obs: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trend": {
                "ema_short": self.trend_ema_short,
                "ema_long": self.trend_ema_long,
                "adx_window": self.trend_adx_window,
                "method": self.trend_method
            },
            "undervaluation": {
                "vwap_window": self.uval_vwap_window,
                "zscore_window": self.uval_zscore_window,
                "dd_lookback": self.uval_dd_lookback,
                "method": self.uval_method
            },
            "hurst": {
                "window": self.hurst_window,
                "method": self.hurst_method
            },
            "volatility": {
                "window": self.vol_window,
                "pct_lookback": self.vol_pct_lookback
            },
            "liquidity": {
                "window": self.liq_window,
                "pct_lookback": self.liq_pct_lookback
            },
            "coupling": {
                "window": self.coupling_window,
                "method": self.coupling_method
            },
            "regime": {
                "n_states": self.regime_n_states,
                "window": self.regime_window,
                "seed": self.regime_seed
            },
            "min_obs": self.min_obs
        }


@dataclass
class IndicatorResult:
    """Result container for all computed indicators."""
    
    # Component scores
    T_t: np.ndarray  # Trend
    U_t: np.ndarray  # Undervaluation
    H_t: np.ndarray  # Hurst
    V_t: np.ndarray  # Volatility regime
    L_t: np.ndarray  # Liquidity
    C_t: np.ndarray  # Coupling
    R_t: np.ndarray  # Regime probability
    
    # Raw values (for debugging)
    raw: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Metadata
    meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Index/dates
    index: Optional[pd.DatetimeIndex] = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with all components."""
        data = {
            'T_t': self.T_t,
            'U_t': self.U_t,
            'H_t': self.H_t,
            'V_t': self.V_t,
            'L_t': self.L_t,
            'C_t': self.C_t,
            'R_t': self.R_t
        }
        
        df = pd.DataFrame(data, index=self.index)
        return df
    
    def get_valid_mask(self) -> np.ndarray:
        """Get mask where all components are valid (not NaN)."""
        return (
            ~np.isnan(self.T_t) &
            ~np.isnan(self.U_t) &
            ~np.isnan(self.H_t) &
            ~np.isnan(self.V_t) &
            ~np.isnan(self.L_t) &
            ~np.isnan(self.C_t) &
            ~np.isnan(self.R_t)
        )


class IndicatorEngine:
    """
    Unified indicator computation engine.
    
    Computes all indicator components from raw OHLCV data.
    
    Example
    -------
    >>> engine = IndicatorEngine()
    >>> result = engine.compute(df)  # df has OHLCV columns
    >>> print(result.T_t[-1], result.U_t[-1])  # Latest values
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        self.config = config or IndicatorConfig()
    
    def compute(
        self,
        df: pd.DataFrame,
        market_data: Optional[pd.DataFrame] = None,
        peer_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> IndicatorResult:
        """
        Compute all indicators from OHLCV data.
        
        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data with columns: Open, High, Low, Close, Volume
        market_data : pd.DataFrame, optional
            Market/benchmark data for coupling calculation
        peer_data : Dict[str, pd.DataFrame], optional
            Peer asset data for coupling calculation
        
        Returns
        -------
        IndicatorResult
            Container with all computed indicator components
        """
        # Extract arrays
        close = df['Close'].values
        high = df['High'].values if 'High' in df.columns else None
        low = df['Low'].values if 'Low' in df.columns else None
        volume = df['Volume'].values if 'Volume' in df.columns else None
        
        n = len(close)
        meta = {}
        raw = {}
        
        # 1. Trend Strength (T_t)
        logger.info("Computing trend strength (T_t)...")
        T_t, meta['trend'] = trend_strength_score(
            close,
            high=high,
            low=low,
            method=self.config.trend_method,
            ema_short=self.config.trend_ema_short,
            ema_long=self.config.trend_ema_long,
            adx_window=self.config.trend_adx_window
        )
        
        # 2. Undervaluation (U_t)
        logger.info("Computing undervaluation (U_t)...")
        U_t, meta['undervaluation'] = undervaluation_score(
            close,
            volumes=volume,
            high=high,
            low=low,
            method=self.config.uval_method,
            vwap_window=self.config.uval_vwap_window,
            zscore_window=self.config.uval_zscore_window,
            dd_lookback=self.config.uval_dd_lookback
        )
        
        # 3. Hurst Exponent (H_t)
        logger.info("Computing Hurst exponent (H_t)...")
        H_t_raw, meta['hurst'] = estimate_hurst(
            close,
            window=self.config.hurst_window,
            method=self.config.hurst_method
        )
        # Hurst is already in [0, 1] range
        H_t = H_t_raw
        raw['hurst_raw'] = H_t_raw
        
        # 4. Volatility Regime (V_t)
        logger.info("Computing volatility regime (V_t)...")
        V_t, meta['volatility'] = volatility_regime_score(
            close,
            vol_window=self.config.vol_window,
            pct_lookback=self.config.vol_pct_lookback
        )
        
        # 5. Liquidity (L_t)
        logger.info("Computing liquidity (L_t)...")
        L_t, meta['liquidity'] = liquidity_score(
            close,
            volume_series=volume,
            window=self.config.liq_window,
            pct_lookback=self.config.liq_pct_lookback
        )
        
        # 6. Coupling (C_t)
        logger.info("Computing systemic coupling (C_t)...")
        
        # Compute returns for coupling
        safe_close = np.maximum(close, 1e-10)
        returns = np.diff(np.log(safe_close))
        returns = np.insert(returns, 0, 0)
        
        # Market returns
        market_returns = None
        if market_data is not None and 'Close' in market_data.columns:
            market_close = market_data['Close'].values
            safe_market = np.maximum(market_close, 1e-10)
            market_returns = np.diff(np.log(safe_market))
            market_returns = np.insert(market_returns, 0, 0)
            # Align lengths
            if len(market_returns) != n:
                market_returns = None
        
        # Peer returns
        other_returns = None
        if peer_data:
            peer_returns_list = []
            for symbol, peer_df in peer_data.items():
                if 'Close' in peer_df.columns and len(peer_df) == n:
                    peer_close = peer_df['Close'].values
                    safe_peer = np.maximum(peer_close, 1e-10)
                    peer_ret = np.diff(np.log(safe_peer))
                    peer_ret = np.insert(peer_ret, 0, 0)
                    peer_returns_list.append(peer_ret)
            
            if peer_returns_list:
                other_returns = np.column_stack(peer_returns_list)
        
        C_t, meta['coupling'] = coupling_score(
            returns,
            market_returns=market_returns,
            other_assets_returns=other_returns,
            window=self.config.coupling_window,
            method=self.config.coupling_method
        )
        
        # 7. Regime Probability (R_t)
        logger.info("Computing regime probability (R_t)...")
        regime_config = HMMRegimeConfig(
            n_states=self.config.regime_n_states,
            window=self.config.regime_window,
            seed=self.config.regime_seed
        )
        R_t, meta['regime'] = infer_regime_prob(close, regime_config)
        
        logger.info("Indicator computation complete.")
        
        return IndicatorResult(
            T_t=T_t,
            U_t=U_t,
            H_t=H_t,
            V_t=V_t,
            L_t=L_t,
            C_t=C_t,
            R_t=R_t,
            raw=raw,
            meta=meta,
            index=df.index if isinstance(df.index, pd.DatetimeIndex) else None
        )
    
    def compute_from_arrays(
        self,
        close: np.ndarray,
        high: Optional[np.ndarray] = None,
        low: Optional[np.ndarray] = None,
        volume: Optional[np.ndarray] = None,
        market_returns: Optional[np.ndarray] = None
    ) -> IndicatorResult:
        """
        Compute indicators from raw arrays.
        
        Convenience method when you don't have a DataFrame.
        """
        df = pd.DataFrame({
            'Close': close
        })
        
        if high is not None:
            df['High'] = high
        if low is not None:
            df['Low'] = low
        if volume is not None:
            df['Volume'] = volume
        
        market_data = None
        if market_returns is not None:
            # Convert returns to synthetic prices
            market_prices = np.cumprod(1 + market_returns)
            market_data = pd.DataFrame({'Close': market_prices})
        
        return self.compute(df, market_data=market_data)


def compute_all_indicators(
    df: pd.DataFrame,
    config: Optional[IndicatorConfig] = None,
    **kwargs
) -> IndicatorResult:
    """
    Convenience function to compute all indicators.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data
    config : IndicatorConfig, optional
        Configuration object
    **kwargs : dict
        Additional arguments passed to IndicatorEngine.compute()
    
    Returns
    -------
    IndicatorResult
        All computed indicators
    """
    engine = IndicatorEngine(config)
    return engine.compute(df, **kwargs)
