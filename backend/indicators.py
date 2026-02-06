import pandas as pd
import numpy as np
from typing import Dict, Optional

# PHASE1: indicator hook - Import Phase 1 advanced indicators
# from indicators import (
#     hurst, hmm_regime, vwap_z, volatility, liquidity, coupling,
#     normalization, committee, composite
# )

class TechnicalIndicators:
    """Calculate technical indicators from price data"""
    
    @staticmethod
    def calculate_sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """Bollinger Bands"""
        middle = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def calculate_atr_percentile(atr: pd.Series, lookback: int = 252) -> pd.Series:
        """ATR Percentile (rolling)"""
        return atr.rolling(window=lookback).apply(lambda x: pd.Series(x).rank().iloc[-1] / len(x) * 100)
    
    @staticmethod
    def calculate_z_score(data: pd.Series, period: int) -> pd.Series:
        """Z-Score (standardized deviation from mean)"""
        rolling_mean = data.rolling(window=period).mean()
        rolling_std = data.rolling(window=period).std()
        z_score = (data - rolling_mean) / rolling_std
        return z_score
    
    @staticmethod
    def calculate_drawdown(data: pd.Series) -> pd.Series:
        """Drawdown percentage from rolling high"""
        rolling_max = data.expanding().max()
        drawdown = ((data - rolling_max) / rolling_max) * 100
        return drawdown
    
    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average Directional Index (trend strength)"""
        # Calculate +DM and -DM
        high_diff = high.diff()
        low_diff = -low.diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        # Calculate ATR
        tr_high_low = high - low
        tr_high_close = np.abs(high - close.shift())
        tr_low_close = np.abs(low - close.shift())
        true_range = pd.concat([tr_high_low, tr_high_close, tr_low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # Calculate DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    @classmethod
    def calculate_all_indicators(cls, df: pd.DataFrame) -> Dict[str, Optional[float]]:
        """Calculate all indicators for the most recent data point"""
        if len(df) < 200:
            return {}
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        # Moving averages
        sma_50 = cls.calculate_sma(close, 50).iloc[-1]
        sma_200 = cls.calculate_sma(close, 200).iloc[-1]
        ema_50 = cls.calculate_ema(close, 50).iloc[-1]
        
        # RSI
        rsi_14 = cls.calculate_rsi(close, 14).iloc[-1]
        
        # MACD
        macd_data = cls.calculate_macd(close)
        macd = macd_data['macd'].iloc[-1]
        macd_signal = macd_data['signal'].iloc[-1]
        macd_hist = macd_data['histogram'].iloc[-1]
        
        # Bollinger Bands
        bb = cls.calculate_bollinger_bands(close)
        bb_upper = bb['upper'].iloc[-1]
        bb_middle = bb['middle'].iloc[-1]
        bb_lower = bb['lower'].iloc[-1]
        
        # ATR
        atr_14 = cls.calculate_atr(high, low, close, 14).iloc[-1]
        atr_series = cls.calculate_atr(high, low, close, 14)
        atr_percentile = cls.calculate_atr_percentile(atr_series, 252).iloc[-1]
        
        # Z-scores
        z_score_20 = cls.calculate_z_score(close, 20).iloc[-1]
        z_score_50 = cls.calculate_z_score(close, 50).iloc[-1]
        z_score_100 = cls.calculate_z_score(close, 100).iloc[-1]
        
        # Drawdown
        drawdown_pct = cls.calculate_drawdown(close).iloc[-1]
        
        # ADX
        adx_14 = cls.calculate_adx(high, low, close, 14).iloc[-1]
        
        return {
            'sma_50': float(sma_50) if not np.isnan(sma_50) else None,
            'sma_200': float(sma_200) if not np.isnan(sma_200) else None,
            'ema_50': float(ema_50) if not np.isnan(ema_50) else None,
            'rsi_14': float(rsi_14) if not np.isnan(rsi_14) else None,
            'macd': float(macd) if not np.isnan(macd) else None,
            'macd_signal': float(macd_signal) if not np.isnan(macd_signal) else None,
            'macd_hist': float(macd_hist) if not np.isnan(macd_hist) else None,
            'bb_upper': float(bb_upper) if not np.isnan(bb_upper) else None,
            'bb_middle': float(bb_middle) if not np.isnan(bb_middle) else None,
            'bb_lower': float(bb_lower) if not np.isnan(bb_lower) else None,
            'atr_14': float(atr_14) if not np.isnan(atr_14) else None,
            'atr_percentile': float(atr_percentile) if not np.isnan(atr_percentile) else None,
            'z_score_20': float(z_score_20) if not np.isnan(z_score_20) else None,
            'z_score_50': float(z_score_50) if not np.isnan(z_score_50) else None,
            'z_score_100': float(z_score_100) if not np.isnan(z_score_100) else None,
            'drawdown_pct': float(drawdown_pct) if not np.isnan(drawdown_pct) else None,
            'adx_14': float(adx_14) if not np.isnan(adx_14) else None
        }
