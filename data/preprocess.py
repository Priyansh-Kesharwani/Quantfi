"""
Data Preprocessing Module

Transforms raw OHLCV data into features required for indicator computation.

Features:
- VWAP approximation from daily data
- Log returns
- Rolling statistics (mean, std)
- Drawdown series
- Missing data handling

Author: Phase 2 Implementation
Date: 2026-02-07
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def preprocess_ohlcv(
    df: pd.DataFrame,
    fill_method: str = "ffill"
) -> pd.DataFrame:
    """
    Preprocess raw OHLCV data.
    
    Steps:
    1. Ensure required columns exist
    2. Handle missing values
    3. Ensure proper index (datetime)
    4. Sort by date
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV data with columns: Open, High, Low, Close, Volume
    fill_method : str
        Method for filling missing values ('ffill', 'bfill', 'drop')
    
    Returns
    -------
    pd.DataFrame
        Preprocessed OHLCV data
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty")
    
    # Create copy to avoid modifying original
    df = df.copy()
    
    # Standardize column names (handle lowercase)
    col_map = {
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'volume': 'Volume', 'adj close': 'Adj Close'
    }
    df.columns = [col_map.get(c.lower(), c) for c in df.columns]
    
    # Ensure required columns
    required = ['Open', 'High', 'Low', 'Close']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Handle Volume if missing
    if 'Volume' not in df.columns:
        logger.warning("Volume column missing, filling with zeros")
        df['Volume'] = 0
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Sort by date
    df = df.sort_index()
    
    # Handle missing values
    if fill_method == "ffill":
        df = df.ffill()
    elif fill_method == "bfill":
        df = df.bfill()
    elif fill_method == "drop":
        df = df.dropna()
    
    # Drop any remaining NaN in price columns
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
    
    return df


def compute_derived_features(
    df: pd.DataFrame,
    windows: Dict[str, int] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compute derived features from OHLCV data.
    
    Features Computed:
    - returns: Simple returns
    - log_returns: Log returns
    - vwap: Approximated VWAP (typical price weighted by volume)
    - typical_price: (H + L + C) / 3
    - rolling_mean_{window}: Rolling mean of close
    - rolling_std_{window}: Rolling standard deviation
    - rolling_vol_{window}: Rolling realized volatility
    - drawdown: Current drawdown from rolling max
    - volume_ma_{window}: Rolling mean of volume
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed OHLCV data
    windows : Dict[str, int]
        Window sizes for rolling calculations
        Default: {"short": 20, "medium": 50, "long": 200, "vol": 20}
    
    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        - DataFrame with derived features
        - Metadata about computed features
    """
    if windows is None:
        windows = {
            "short": 20,
            "medium": 50,
            "long": 200,
            "vol": 20,
            "vwap": 20
        }
    
    df = df.copy()
    meta = {
        "windows": windows,
        "features_computed": []
    }
    
    # Simple returns
    df['returns'] = df['Close'].pct_change()
    meta["features_computed"].append("returns")
    
    # Log returns
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    meta["features_computed"].append("log_returns")
    
    # Typical price: (H + L + C) / 3
    df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
    meta["features_computed"].append("typical_price")
    
    # VWAP approximation (rolling typical price weighted by volume)
    # Note: True intraday VWAP not available from daily data
    vwap_window = windows.get("vwap", 20)
    if df['Volume'].sum() > 0:
        df['vwap'] = (
            (df['typical_price'] * df['Volume']).rolling(vwap_window).sum() /
            df['Volume'].rolling(vwap_window).sum()
        )
    else:
        # Fallback to simple moving average of typical price
        df['vwap'] = df['typical_price'].rolling(vwap_window).mean()
    meta["features_computed"].append("vwap")
    
    # Price relative to VWAP
    df['price_vwap_ratio'] = df['Close'] / df['vwap']
    meta["features_computed"].append("price_vwap_ratio")
    
    # Rolling statistics
    for name, window in windows.items():
        if name in ["vwap"]:
            continue
        
        # Rolling mean of close
        col_mean = f'rolling_mean_{name}'
        df[col_mean] = df['Close'].rolling(window).mean()
        meta["features_computed"].append(col_mean)
        
        # Rolling standard deviation
        col_std = f'rolling_std_{name}'
        df[col_std] = df['Close'].rolling(window).std()
        meta["features_computed"].append(col_std)
    
    # Realized volatility (annualized)
    vol_window = windows.get("vol", 20)
    df['realized_vol'] = df['log_returns'].rolling(vol_window).std() * np.sqrt(252)
    meta["features_computed"].append("realized_vol")
    
    # Drawdown from rolling max
    rolling_max = df['Close'].expanding().max()
    df['drawdown'] = (df['Close'] - rolling_max) / rolling_max
    meta["features_computed"].append("drawdown")
    
    # Volume moving average
    vol_ma_window = windows.get("short", 20)
    df['volume_ma'] = df['Volume'].rolling(vol_ma_window).mean()
    meta["features_computed"].append("volume_ma")
    
    # Volume ratio (current / average)
    df['volume_ratio'] = df['Volume'] / df['volume_ma'].replace(0, np.nan)
    meta["features_computed"].append("volume_ratio")
    
    # Price momentum (rate of change)
    for name, window in [("short", windows.get("short", 20)), 
                         ("medium", windows.get("medium", 50))]:
        col = f'momentum_{name}'
        df[col] = df['Close'].pct_change(window)
        meta["features_computed"].append(col)
    
    # True Range for ATR calculation
    df['tr'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            np.abs(df['High'] - df['Close'].shift(1)),
            np.abs(df['Low'] - df['Close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(windows.get("short", 20)).mean()
    meta["features_computed"].append("tr")
    meta["features_computed"].append("atr")
    
    return df, meta


def align_multiple_series(
    data: Dict[str, pd.DataFrame],
    column: str = 'Close'
) -> pd.DataFrame:
    """
    Align multiple price series to common dates.
    
    Parameters
    ----------
    data : Dict[str, pd.DataFrame]
        Dict of symbol -> DataFrame
    column : str
        Column to extract (default: 'Close')
    
    Returns
    -------
    pd.DataFrame
        DataFrame with symbols as columns, aligned on dates
    """
    series = {}
    
    for symbol, df in data.items():
        if df is not None and column in df.columns:
            series[symbol] = df[column]
    
    if not series:
        return pd.DataFrame()
    
    aligned = pd.DataFrame(series)
    
    # Forward fill missing values (up to 5 days)
    aligned = aligned.ffill(limit=5)
    
    return aligned


def compute_rolling_correlation(
    df: pd.DataFrame,
    target: str,
    window: int = 60
) -> pd.DataFrame:
    """
    Compute rolling correlation of target with other columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Aligned price data (symbols as columns)
    target : str
        Target symbol for correlation
    window : int
        Rolling window size (default: 60 days)
    
    Returns
    -------
    pd.DataFrame
        Rolling correlations with each peer
    """
    if target not in df.columns:
        raise ValueError(f"Target {target} not in DataFrame")
    
    # Compute returns for correlation
    returns = df.pct_change().dropna()
    
    if returns.empty:
        return pd.DataFrame()
    
    correlations = pd.DataFrame(index=returns.index)
    
    target_returns = returns[target]
    
    for col in returns.columns:
        if col != target:
            correlations[f'corr_{col}'] = (
                target_returns.rolling(window).corr(returns[col])
            )
    
    return correlations
