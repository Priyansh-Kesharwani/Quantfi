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
    pass


def timeout_wrapper(timeout_seconds: int):
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
    
    SYMBOL_MAP = {
        'XAU': 'GC=F',                  
        'XAG': 'SI=F',                    
        'GOLD': 'GC=F',
        'SILVER': 'SI=F',
        'NIFTY': '^NSEI',
        'NIFTY50': '^NSEI',
        'SENSEX': '^BSESN',
        'SILVERIETF': 'SILVERBEES.NS',
        'SILVERETF': 'SILVERBEES.NS',
        'SILVERBEES': 'SILVERBEES.NS',
        'GOLDIETF': 'GOLDBEES.NS',
        'GOLDETF': 'GOLDBEES.NS',
        'GOLDBEES': 'GOLDBEES.NS',
    }
    
    INDIAN_SUFFIX = '.NS'
    
    MAJOR_NSE_STOCKS = [
        'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 
        'ICICIBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS',
        'KOTAKBANK.NS', 'LT.NS'
    ]
    
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
        
        from engine.utils.data_cache import DataCache
        self.cache = DataCache(cache_dir=cache_dir)
        
        try:
            import yfinance as yf
            self._yf = yf
        except ImportError:
            logger.warning("yfinance not available - will use mock data")
            self._yf = None
    
    def _resolve_symbol(self, symbol: str) -> str:
        symbol_upper = symbol.upper()
        
        if symbol_upper in self.SYMBOL_MAP:
            return self.SYMBOL_MAP[symbol_upper]
        
        if '.' in symbol or '=' in symbol or symbol.startswith('^'):
            return symbol
        
        return symbol
    
    @timeout_wrapper(30)
    def _fetch_daily_raw(self, symbol: str, period: str = "max") -> Optional[pd.DataFrame]:
        if self._yf is None:
            logger.error(f"yfinance not available — cannot fetch real data for {symbol}")
            return None
        
        provider_symbol = self._resolve_symbol(symbol)
        
        try:
            ticker = self._yf.Ticker(provider_symbol)
            df = ticker.history(period=period)
            
            if df.empty:
                logger.warning(f"No real data returned for {symbol} (period={period})")
                return None
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None
    
    
    def fetch_daily(
        self, 
        symbol: str, 
        period: str = "max",
        use_cache: bool = True
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
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
        
        if use_cache:
            cached = self.cache.get(symbol, interval="1d")
            if cached is not None:
                meta["source"] = "cache"
                meta["rows"] = len(cached)
                meta["status"] = "success"
                meta["elapsed_ms"] = int((time.time() - start_time) * 1000)
                return cached, meta
        
        df = None
        for attempt in range(self.max_retries + 1):
            try:
                df = self._fetch_daily_raw(symbol, period)
                
                if df is not None and not df.empty:
                    meta["source"] = "api"
                    meta["rows"] = len(df)
                    meta["status"] = "success"
                    
                    if use_cache:
                        self.cache.put(symbol, df, interval="1d")
                    
                    break
                    
            except TimeoutError as e:
                meta["error"] = f"Timeout on attempt {attempt + 1}"
                logger.warning(f"Timeout fetching {symbol} (attempt {attempt + 1})")
                
            except Exception as e:
                meta["error"] = str(e)
                logger.warning(f"Error fetching {symbol}: {e} (attempt {attempt + 1})")
            
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
        resolved = self._resolve_symbol(symbol)
        
        if resolved.endswith('.NS'):
            peers = self.MAJOR_NSE_STOCKS[:n_peers]
        elif resolved in ['GC=F', 'SI=F']:
            peers = ['GC=F', 'SI=F', 'SPY', 'TLT', 'DX-Y.NYB'][:n_peers]
        else:
            peers = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD'][:n_peers]
        
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
        df, _ = self.fetch_daily(symbol, period="max")
        
        if df is None or df.empty:
            return None
        
        start = df.index.min()
        end = df.index.max()
        years = (end - start).days / 365.25
        
        return round(years, 1)
