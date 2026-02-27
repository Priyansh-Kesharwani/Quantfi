import pandas as pd
import numpy as np
from typing import Dict, Optional
from backend.app_config import get_backend_config

CFG = get_backend_config()


class TechnicalIndicators:
    
    @staticmethod
    def calculate_sma(data: pd.Series, period: int) -> pd.Series:
        return data.rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(data: pd.Series, period: Optional[int] = None) -> pd.Series:
        if period is None:
            period = CFG.rsi_period
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(
        data: pd.Series,
        fast: Optional[int] = None,
        slow: Optional[int] = None,
        signal: Optional[int] = None,
    ) -> Dict[str, pd.Series]:
        fast = fast or CFG.macd_fast
        slow = slow or CFG.macd_slow
        signal = signal or CFG.macd_signal
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
    def calculate_bollinger_bands(
        data: pd.Series,
        period: Optional[int] = None,
        std_dev: Optional[float] = None,
    ) -> Dict[str, pd.Series]:
        period = period or CFG.bollinger_period
        std_dev = std_dev or CFG.bollinger_std_dev
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
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: Optional[int] = None) -> pd.Series:
        period = period or CFG.atr_period
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def calculate_atr_percentile(atr: pd.Series, lookback: Optional[int] = None) -> pd.Series:
        lookback = lookback or CFG.atr_percentile_lookback
        return atr.rolling(window=lookback).apply(lambda x: pd.Series(x).rank().iloc[-1] / len(x) * 100)
    
    @staticmethod
    def calculate_z_score(data: pd.Series, period: int) -> pd.Series:
        rolling_mean = data.rolling(window=period).mean()
        rolling_std = data.rolling(window=period).std()
        z_score = (data - rolling_mean) / rolling_std
        return z_score
    
    @staticmethod
    def calculate_drawdown(data: pd.Series) -> pd.Series:
        rolling_max = data.expanding().max()
        drawdown = ((data - rolling_max) / rolling_max) * 100
        return drawdown
    
    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: Optional[int] = None) -> pd.Series:
        period = period or CFG.adx_period
        high_diff = high.diff()
        low_diff = -low.diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        tr_high_low = high - low
        tr_high_close = np.abs(high - close.shift())
        tr_low_close = np.abs(low - close.shift())
        true_range = pd.concat([tr_high_low, tr_high_close, tr_low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    @classmethod
    def calculate_all_indicators(cls, df: pd.DataFrame) -> Dict[str, Optional[float]]:
        if len(df) < CFG.indicator_min_history_rows:
            return {}
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        sma_50 = cls.calculate_sma(close, CFG.sma_short_period).iloc[-1]
        sma_200 = cls.calculate_sma(close, CFG.sma_long_period).iloc[-1]
        ema_50 = cls.calculate_ema(close, CFG.ema_period).iloc[-1]
        
        rsi_14 = cls.calculate_rsi(close).iloc[-1]
        
        macd_data = cls.calculate_macd(close)
        macd = macd_data['macd'].iloc[-1]
        macd_signal = macd_data['signal'].iloc[-1]
        macd_hist = macd_data['histogram'].iloc[-1]
        
        bb = cls.calculate_bollinger_bands(close)
        bb_upper = bb['upper'].iloc[-1]
        bb_middle = bb['middle'].iloc[-1]
        bb_lower = bb['lower'].iloc[-1]
        
        atr_14 = cls.calculate_atr(high, low, close).iloc[-1]
        atr_series = cls.calculate_atr(high, low, close)
        atr_percentile = cls.calculate_atr_percentile(atr_series).iloc[-1]
        
        z0, z1, z2 = CFG.z_score_periods
        z_score_20 = cls.calculate_z_score(close, z0).iloc[-1]
        z_score_50 = cls.calculate_z_score(close, z1).iloc[-1]
        z_score_100 = cls.calculate_z_score(close, z2).iloc[-1]
        
        drawdown_pct = cls.calculate_drawdown(close).iloc[-1]
        
        adx_14 = cls.calculate_adx(high, low, close).iloc[-1]
        
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
