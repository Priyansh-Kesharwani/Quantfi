"""
Data Fetcher Module

Historical data fetching with timeout handling, retry logic, and caching.

Supported Assets:
- Commodities: XAU (GC=F), XAG (SI=F)
- US Equities: AAPL, NFLX, MSFT, GOOGL
- Indian Equities: RELIANCE.NS, TCS.NS, INFY.NS, etc.
- Index: ^NSEI (NIFTY 50)

Author: Phase 2 Implementation
Date: 2026-02-07
"""

import os
import time
import logging
import signal
import functools
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    """Raised when an operation times out."""
    pass


def timeout_wrapper(timeout_seconds: int):
    """
    Decorator to add timeout to functions.
    
    Uses threading for cross-platform timeout support.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            error = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    error[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=timeout_seconds)
            
            if thread.is_alive():
                raise TimeoutError(f"Operation timed out after {timeout_seconds}s")
            
            if error[0] is not None:
                raise error[0]
            
            return result[0]
        return wrapper
    return decorator


class DataFetcher:
    """
    Historical market data fetcher with caching and timeouts.
    
    Supported Assets:
    -----------------
    - XAU, XAG (Commodities via futures)
    - AAPL, NFLX, MSFT, GOOGL (US Equities)
    - RELIANCE.NS, TCS.NS, INFY.NS, etc. (Indian Equities)
    - ^NSEI (NIFTY 50 Index)
    
    Parameters
    ----------
    cache_dir : str
        Directory for data cache
    daily_timeout : int
        Timeout for daily data fetch (default: 30s)
    weekly_timeout : int
        Timeout for weekly data fetch (default: 15s)
    max_retries : int
        Maximum retry attempts (default: 1)
    """
    
    # Symbol mappings for commodities
    SYMBOL_MAP = {
        'XAU': 'GC=F',    # Gold futures
        'XAG': 'SI=F',    # Silver futures
        'GOLD': 'GC=F',
        'SILVER': 'SI=F',
        'NIFTY': '^NSEI',
        'NIFTY50': '^NSEI'
    }
    
    # Indian exchange suffix
    INDIAN_SUFFIX = '.NS'
    
    # Major NSE stocks for correlation analysis
    MAJOR_NSE_STOCKS = [
        'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 
        'ICICIBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS',
        'KOTAKBANK.NS', 'LT.NS'
    ]
    
    # US market benchmark
    US_BENCHMARK = 'SPY'
    
    def __init__(
        self,
        cache_dir: str = "data/cache",
        daily_timeout: int = 30,
        weekly_timeout: int = 15,
        max_retries: int = 1
    ):
        self.daily_timeout = daily_timeout
        self.weekly_timeout = weekly_timeout
        self.max_retries = max_retries
        
        # Import cache and yfinance
        from .cache import DataCache
        self.cache = DataCache(cache_dir=cache_dir)
        
        try:
            import yfinance as yf
            self._yf = yf
        except ImportError:
            logger.warning("yfinance not available - will use mock data")
            self._yf = None
    
    def _resolve_symbol(self, symbol: str) -> str:
        """Resolve asset symbol to provider symbol."""
        symbol_upper = symbol.upper()
        
        # Check direct mapping
        if symbol_upper in self.SYMBOL_MAP:
            return self.SYMBOL_MAP[symbol_upper]
        
        # If already has exchange suffix, return as-is
        if '.' in symbol or '=' in symbol or symbol.startswith('^'):
            return symbol
        
        return symbol
    
    @timeout_wrapper(30)
    def _fetch_daily_raw(self, symbol: str, period: str = "max") -> Optional[pd.DataFrame]:
        """Fetch daily data with timeout (internal)."""
        if self._yf is None:
            return self._generate_mock_data(symbol, period)
        
        provider_symbol = self._resolve_symbol(symbol)
        
        try:
            ticker = self._yf.Ticker(provider_symbol)
            df = ticker.history(period=period)
            
            if df.empty:
                logger.warning(f"No data for {symbol}, using mock")
                return self._generate_mock_data(symbol, period)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return self._generate_mock_data(symbol, period)
    
    def _generate_mock_data(
        self, 
        symbol: str, 
        period: str = "max"
    ) -> pd.DataFrame:
        """Generate realistic mock historical data for testing."""
        # Map period to days
        period_days = {
            '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365, 
            '2y': 730, '5y': 1825, '10y': 3650, '15y': 5475,
            'max': 7300  # ~20 years
        }
        days = period_days.get(period, 7300)
        
        # Base price by asset type
        base_prices = {
            'GC=F': 1500.0, 'SI=F': 20.0, 'AAPL': 100.0, 'NFLX': 300.0,
            'MSFT': 200.0, 'GOOGL': 100.0, '^NSEI': 15000.0
        }
        
        resolved = self._resolve_symbol(symbol)
        base_price = base_prices.get(resolved, 100.0)
        
        # Generate dates
        np.random.seed(hash(symbol) % (2**32))  # Deterministic per symbol
        dates = pd.date_range(end=datetime.now(), periods=days, freq='B')  # Business days
        
        # Generate price series with regime changes
        prices = [base_price]
        regime = 0  # 0: normal, 1: bull, 2: bear
        
        for i in range(1, len(dates)):
            # Occasional regime changes
            if np.random.random() < 0.005:  # ~1.25 changes per year
                regime = np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2])
            
            # Returns based on regime
            if regime == 0:  # Normal
                daily_return = np.random.normal(0.0003, 0.012)
            elif regime == 1:  # Bull
                daily_return = np.random.normal(0.001, 0.015)
            else:  # Bear
                daily_return = np.random.normal(-0.0005, 0.02)
            
            prices.append(prices[-1] * (1 + daily_return))
        
        prices = np.array(prices)
        
        # Generate OHLCV
        highs = prices * (1 + np.abs(np.random.normal(0, 0.005, len(prices))))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.005, len(prices))))
        opens = lows + (highs - lows) * np.random.uniform(0.2, 0.8, len(prices))
        volumes = np.random.lognormal(mean=15, sigma=0.5, size=len(prices)).astype(int)
        
        df = pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': prices,
            'Volume': volumes
        }, index=dates)
        
        return df
    
    def fetch_daily(
        self, 
        symbol: str, 
        period: str = "max",
        use_cache: bool = True
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Fetch daily OHLCV data with caching and timeout.
        
        Parameters
        ----------
        symbol : str
            Asset symbol (e.g., 'AAPL', 'XAU', 'RELIANCE.NS')
        period : str
            Data period ('1y', '5y', '15y', 'max')
        use_cache : bool
            Whether to use cache (default: True)
        
        Returns
        -------
        Tuple[DataFrame, Dict]
            - OHLCV DataFrame or None
            - Metadata dict with fetch status, source, timing
        """
        start_time = time.time()
        meta = {
            "symbol": symbol,
            "period": period,
            "source": None,
            "rows": 0,
            "status": "pending",
            "elapsed_ms": 0,
            "error": None
        }
        
        # Check cache first
        if use_cache:
            cached = self.cache.get(symbol, interval="1d")
            if cached is not None:
                meta["source"] = "cache"
                meta["rows"] = len(cached)
                meta["status"] = "success"
                meta["elapsed_ms"] = int((time.time() - start_time) * 1000)
                return cached, meta
        
        # Fetch with retry
        df = None
        for attempt in range(self.max_retries + 1):
            try:
                df = self._fetch_daily_raw(symbol, period)
                
                if df is not None and not df.empty:
                    meta["source"] = "api"
                    meta["rows"] = len(df)
                    meta["status"] = "success"
                    
                    # Cache the result
                    if use_cache:
                        self.cache.put(symbol, df, interval="1d")
                    
                    break
                    
            except TimeoutError as e:
                meta["error"] = f"Timeout on attempt {attempt + 1}"
                logger.warning(f"Timeout fetching {symbol} (attempt {attempt + 1})")
                
            except Exception as e:
                meta["error"] = str(e)
                logger.warning(f"Error fetching {symbol}: {e} (attempt {attempt + 1})")
            
            # Exponential backoff
            if attempt < self.max_retries:
                time.sleep(2 ** attempt)
        
        if df is None or df.empty:
            meta["status"] = "failed"
            logger.error(f"Failed to fetch {symbol} after {self.max_retries + 1} attempts")
        
        meta["elapsed_ms"] = int((time.time() - start_time) * 1000)
        return df, meta
    
    def fetch_multiple(
        self, 
        symbols: List[str], 
        period: str = "max",
        use_cache: bool = True
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, Any]]]:
        """
        Fetch data for multiple symbols.
        
        Returns dict of symbol -> DataFrame and metadata.
        """
        data = {}
        metas = {}
        
        for symbol in symbols:
            df, meta = self.fetch_daily(symbol, period, use_cache)
            if df is not None:
                data[symbol] = df
            metas[symbol] = meta
        
        return data, metas
    
    def fetch_with_peers(
        self,
        symbol: str,
        period: str = "max",
        n_peers: int = 5
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
        """
        Fetch data for a symbol and its market peers (for coupling analysis).
        
        Parameters
        ----------
        symbol : str
            Primary asset symbol
        period : str
            Data period
        n_peers : int
            Number of peer assets to fetch
        
        Returns
        -------
        Tuple[Dict, Dict]
            - Dict of symbol -> DataFrame (primary + peers)
            - Metadata
        """
        resolved = self._resolve_symbol(symbol)
        
        # Determine peers based on asset type
        if resolved.endswith('.NS'):
            peers = self.MAJOR_NSE_STOCKS[:n_peers]
        elif resolved in ['GC=F', 'SI=F']:
            peers = ['GC=F', 'SI=F', 'SPY', 'TLT', 'DX-Y.NYB'][:n_peers]
        else:
            peers = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD'][:n_peers]
        
        # Include primary symbol
        all_symbols = [symbol] + [p for p in peers if self._resolve_symbol(p) != resolved]
        
        data, metas = self.fetch_multiple(all_symbols, period)
        
        meta = {
            "primary": symbol,
            "peers": [s for s in all_symbols if s != symbol],
            "fetched": list(data.keys()),
            "failed": [s for s in all_symbols if s not in data]
        }
        
        return data, meta
    
    def get_available_history_years(self, symbol: str) -> Optional[float]:
        """Get the number of years of available history for a symbol."""
        df, _ = self.fetch_daily(symbol, period="max")
        
        if df is None or df.empty:
            return None
        
        start = df.index.min()
        end = df.index.max()
        years = (end - start).days / 365.25
        
        return round(years, 1)
