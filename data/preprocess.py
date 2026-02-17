import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def preprocess_ohlcv(
    df: pd.DataFrame,
    fill_method: str = "ffill"
) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty")
    
    df = df.copy()
    
    col_map = {
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'volume': 'Volume', 'adj close': 'Adj Close'
    }
    df.columns = [col_map.get(c.lower(), c) for c in df.columns]
    
    required = ['Open', 'High', 'Low', 'Close']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    if 'Volume' not in df.columns:
        logger.warning("Volume column missing, filling with zeros")
        df['Volume'] = 0
    
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    df = df.sort_index()
    
    if fill_method == "ffill":
        df = df.ffill()
    elif fill_method == "bfill":
        df = df.bfill()
    elif fill_method == "drop":
        df = df.dropna()
    
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
    
    return df


def compute_derived_features(
    df: pd.DataFrame,
    windows: Dict[str, int] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
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
    
    df['returns'] = df['Close'].pct_change()
    meta["features_computed"].append("returns")
    
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    meta["features_computed"].append("log_returns")
    
    df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
    meta["features_computed"].append("typical_price")
    
    vwap_window = windows.get("vwap", 20)
    if df['Volume'].sum() > 0:
        df['vwap'] = (
            (df['typical_price'] * df['Volume']).rolling(vwap_window).sum() /
            df['Volume'].rolling(vwap_window).sum()
        )
    else:
        df['vwap'] = df['typical_price'].rolling(vwap_window).mean()
    meta["features_computed"].append("vwap")
    
    df['price_vwap_ratio'] = df['Close'] / df['vwap']
    meta["features_computed"].append("price_vwap_ratio")
    
    for name, window in windows.items():
        if name in ["vwap"]:
            continue
        
        col_mean = f'rolling_mean_{name}'
        df[col_mean] = df['Close'].rolling(window).mean()
        meta["features_computed"].append(col_mean)
        
        col_std = f'rolling_std_{name}'
        df[col_std] = df['Close'].rolling(window).std()
        meta["features_computed"].append(col_std)
    
    vol_window = windows.get("vol", 20)
    df['realized_vol'] = df['log_returns'].rolling(vol_window).std() * np.sqrt(252)
    meta["features_computed"].append("realized_vol")
    
    rolling_max = df['Close'].expanding().max()
    df['drawdown'] = (df['Close'] - rolling_max) / rolling_max
    meta["features_computed"].append("drawdown")
    
    vol_ma_window = windows.get("short", 20)
    df['volume_ma'] = df['Volume'].rolling(vol_ma_window).mean()
    meta["features_computed"].append("volume_ma")
    
    df['volume_ratio'] = df['Volume'] / df['volume_ma'].replace(0, np.nan)
    meta["features_computed"].append("volume_ratio")
    
    for name, window in [("short", windows.get("short", 20)), 
                         ("medium", windows.get("medium", 50))]:
        col = f'momentum_{name}'
        df[col] = df['Close'].pct_change(window)
        meta["features_computed"].append(col)
    
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
    series = {}
    
    for symbol, df in data.items():
        if df is not None and column in df.columns:
            series[symbol] = df[column]
    
    if not series:
        return pd.DataFrame()
    
    aligned = pd.DataFrame(series)
    
    aligned = aligned.ffill(limit=5)
    
    return aligned


def compute_rolling_correlation(
    df: pd.DataFrame,
    target: str,
    window: int = 60
) -> pd.DataFrame:
    if target not in df.columns:
        raise ValueError(f"Target {target} not in DataFrame")
    
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
