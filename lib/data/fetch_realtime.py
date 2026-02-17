import os
import sys
import time
import json
import asyncio
import logging
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class FetchConfig:
    
    symbols: List[str] = field(default_factory=lambda: [
        'GC=F',                           
        'SI=F',                             
        'SPY', 'AAPL', 'NFLX',
        'RELIANCE.NS', 'TCS.NS',
        '^NSEI'                
    ])
    
    symbol_aliases: Dict[str, str] = field(default_factory=lambda: {
        'GC=F': 'XAUUSD',
        'SI=F': 'XAGUSD',
        '^NSEI': 'NIFTY50'
    })
    
    intervals: List[str] = field(default_factory=lambda: ['1d', '1wk', '1h'])
    
    periods: Dict[str, str] = field(default_factory=lambda: {
        '1d': 'max',                                       
        '1wk': 'max',                        
        '1h': '60d',                                     
        '15m': '60d',                   
        '5m': '60d'                     
    })
    
    cache_dir: str = "data_cache"
    
    max_workers: int = 4
    requests_per_minute: int = 30
    backoff_base: float = 2.0
    max_retries: int = 3
    retry_delay_base: float = 1.0
    
    min_data_coverage: float = 0.8                         
    
    alpha_vantage_key: Optional[str] = None
    polygon_key: Optional[str] = None
    
    def __post_init__(self):
        self.alpha_vantage_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
        self.polygon_key = os.environ.get('POLYGON_API_KEY')


@dataclass
class FetchResult:
    symbol: str
    interval: str
    source: str
    status: str                                                 
    data: Optional[pd.DataFrame] = None
    rows: int = 0
    elapsed_ms: int = 0
    retries: int = 0
    rate_limited: bool = False
    error: Optional[str] = None
    cache_path: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'interval': self.interval,
            'source': self.source,
            'status': self.status,
            'rows': self.rows,
            'elapsed_ms': self.elapsed_ms,
            'retries': self.retries,
            'rate_limited': self.rate_limited,
            'error': self.error,
            'cache_path': self.cache_path
        }


class RateLimitHandler:
    
    def __init__(
        self,
        requests_per_minute: int = 30,
        backoff_base: float = 2.0,
        max_retries: int = 3
    ):
        self.requests_per_minute = requests_per_minute
        self.backoff_base = backoff_base
        self.max_retries = max_retries
        
        self._request_times: List[float] = []
        self._lock = threading.Lock()
        self._consecutive_failures = 0
    
    def wait_if_needed(self) -> None:
        with self._lock:
            now = time.time()
            self._request_times = [t for t in self._request_times if now - t < 60]
            
            if len(self._request_times) >= self.requests_per_minute:
                wait_time = 60 - (now - self._request_times[0]) + 0.5
                if wait_time > 0:
                    logger.info(f"Rate limit approaching, waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
    
    def record_request(self) -> None:
        with self._lock:
            self._request_times.append(time.time())
            self._consecutive_failures = 0
    
    def record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
    
    def get_backoff_delay(self, attempt: int) -> float:
        delay = self.backoff_base ** attempt
        jitter = delay * 0.1 * (2 * np.random.random() - 1)
        return delay + jitter
    
    def should_retry(self, attempt: int) -> bool:
        return attempt < self.max_retries
    
    def get_stats(self) -> Dict:
        with self._lock:
            return {
                'requests_last_minute': len(self._request_times),
                'limit': self.requests_per_minute,
                'consecutive_failures': self._consecutive_failures
            }


class DataSource:
    
    name: str = "base"
    
    def fetch(
        self,
        symbol: str,
        interval: str,
        period: str,
        rate_limiter: RateLimitHandler
    ) -> Tuple[Optional[pd.DataFrame], Dict]:
        raise NotImplementedError


class YFinanceSource(DataSource):
    
    name = "yfinance"
    
    def __init__(self):
        try:
            import yfinance as yf
            self._yf = yf
            self._available = True
        except ImportError:
            self._yf = None
            self._available = False
            logger.warning("yfinance not available")
    
    def fetch(
        self,
        symbol: str,
        interval: str,
        period: str,
        rate_limiter: RateLimitHandler
    ) -> Tuple[Optional[pd.DataFrame], Dict]:
        meta = {'source': self.name, 'rate_limited': False, 'retries': 0}
        
        if not self._available:
            return None, {**meta, 'error': 'yfinance not installed'}
        
        for attempt in range(rate_limiter.max_retries + 1):
            try:
                rate_limiter.wait_if_needed()
                
                ticker = self._yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)
                
                rate_limiter.record_request()
                
                if df is not None and not df.empty:
                    df = self._standardize_df(df)
                    return df, meta
                else:
                    meta['error'] = 'No data returned'
                    return None, meta
                    
            except Exception as e:
                error_str = str(e).lower()
                meta['retries'] = attempt + 1
                
                if 'rate' in error_str or 'limit' in error_str or '429' in error_str:
                    meta['rate_limited'] = True
                    rate_limiter.record_failure()
                    
                    if rate_limiter.should_retry(attempt):
                        delay = rate_limiter.get_backoff_delay(attempt)
                        logger.warning(f"Rate limited on {symbol}, waiting {delay:.1f}s")
                        time.sleep(delay)
                        continue
                
                meta['error'] = str(e)
                
                if rate_limiter.should_retry(attempt):
                    delay = rate_limiter.get_backoff_delay(attempt)
                    time.sleep(delay)
                    continue
                    
                return None, meta
        
        return None, meta
    
    def _standardize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        col_map = {
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume'
        }
        df.columns = [col_map.get(c.lower(), c) for c in df.columns]
        
        df = df[~df.index.duplicated(keep='first')]
        
        df = df.sort_index()
        
        return df


class AlphaVantageSource(DataSource):
    
    name = "alpha_vantage"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('ALPHA_VANTAGE_API_KEY')
        self._available = bool(self.api_key)
        
        if not self._available:
            logger.info("Alpha Vantage API key not set")
    
    def fetch(
        self,
        symbol: str,
        interval: str,
        period: str,
        rate_limiter: RateLimitHandler
    ) -> Tuple[Optional[pd.DataFrame], Dict]:
        meta = {'source': self.name, 'rate_limited': False, 'retries': 0}
        
        if not self._available:
            return None, {**meta, 'error': 'API key not configured'}
        
        try:
            import requests
            
            rate_limiter.wait_if_needed()
            
            if interval in ['1d', '1wk']:
                function = 'TIME_SERIES_DAILY_ADJUSTED'
                outputsize = 'full'
            else:
                function = 'TIME_SERIES_INTRADAY'
                outputsize = 'full'
            
            av_symbol = symbol.replace('=F', '').replace('^', '')
            
            params = {
                'function': function,
                'symbol': av_symbol,
                'apikey': self.api_key,
                'outputsize': outputsize
            }
            
            if function == 'TIME_SERIES_INTRADAY':
                params['interval'] = interval.replace('m', 'min').replace('h', '0min')
            
            response = requests.get(
                'https://www.alphavantage.co/query',
                params=params,
                timeout=30
            )
            
            rate_limiter.record_request()
            
            data = response.json()
            
            if 'Note' in data or 'Information' in data:
                meta['rate_limited'] = True
                meta['error'] = data.get('Note') or data.get('Information')
                return None, meta
            
            time_series_key = [k for k in data.keys() if 'Time Series' in k]
            if not time_series_key:
                meta['error'] = 'No time series data in response'
                return None, meta
            
            ts_data = data[time_series_key[0]]
            
            df = pd.DataFrame.from_dict(ts_data, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            df.columns = [c.split('. ')[1] if '. ' in c else c for c in df.columns]
            df = df.rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low',
                'close': 'Close', 'volume': 'Volume',
                'adjusted close': 'Adj Close'
            })
            
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df, meta
            
        except Exception as e:
            meta['error'] = str(e)
            return None, meta


class MockDataSource(DataSource):
    
    name = "mock"
    
    def fetch(
        self,
        symbol: str,
        interval: str,
        period: str,
        rate_limiter: RateLimitHandler
    ) -> Tuple[Optional[pd.DataFrame], Dict]:
        meta = {'source': self.name, 'rate_limited': False, 'retries': 0}
        
        end_date = datetime.now()
        
        period_days = {
            '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365,
            '2y': 730, '5y': 1825, '10y': 3650, '15y': 5475,
            '60d': 60, '90d': 90, 'max': 7300
        }
        days = period_days.get(period, 730)
        
        if interval == '1d':
            freq = 'B'                 
            n_points = days
        elif interval == '1wk':
            freq = 'W'
            n_points = days // 7
        elif interval == '1h':
            freq = 'H'
            n_points = min(days * 8, 60 * 8)                   
        elif interval == '15m':
            freq = '15T'
            n_points = min(days * 32, 60 * 32)
        else:
            freq = 'B'
            n_points = days
        
        dates = pd.date_range(end=end_date, periods=n_points, freq=freq)
        
        np.random.seed(hash(symbol) % (2**32))
        
        base_prices = {
            'GC=F': 1800, 'SI=F': 25, 'SPY': 450, 'AAPL': 180,
            'NFLX': 600, 'RELIANCE.NS': 2500, 'TCS.NS': 3500, '^NSEI': 18000
        }
        base = base_prices.get(symbol, 100)
        
        returns = np.random.normal(0.0003, 0.015, n_points)
        
        regime_changes = np.random.choice(n_points, size=max(1, n_points // 500), replace=False)
        for rc in regime_changes:
            if np.random.random() > 0.5:
                returns[rc:min(rc+20, n_points)] *= 2.5
                returns[rc:min(rc+20, n_points)] -= 0.02
            else:
                returns[rc:min(rc+20, n_points)] *= 1.5
                returns[rc:min(rc+20, n_points)] += 0.01
        
        prices = base * np.cumprod(1 + returns)
        
        highs = prices * (1 + np.abs(np.random.normal(0, 0.005, n_points)))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.005, n_points)))
        opens = lows + (highs - lows) * np.random.uniform(0.2, 0.8, n_points)
        volumes = np.random.lognormal(mean=15, sigma=0.5, size=n_points).astype(int)
        
        df = pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': prices,
            'Volume': volumes
        }, index=dates)
        
        return df, meta


class CacheManager:
    
    def __init__(self, cache_dir: str = "data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_path = self.cache_dir / "fetch_metadata.json"
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        if self._metadata_path.exists():
            try:
                with open(self._metadata_path, 'r') as f:
                    self._metadata = json.load(f)
            except:
                self._metadata = {}
        else:
            self._metadata = {}
    
    def _save_metadata(self) -> None:
        with open(self._metadata_path, 'w') as f:
            json.dump(self._metadata, f, indent=2, default=str)
    
    def _cache_key(self, symbol: str, interval: str) -> str:
        safe_symbol = symbol.replace('=', '_').replace('^', '_')
        return f"{safe_symbol}_{interval}"
    
    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.csv"
    
    def is_cached(self, symbol: str, interval: str, max_age_hours: int = 24) -> bool:
        key = self._cache_key(symbol, interval)
        
        if key not in self._metadata:
            return False
        
        cached_time = datetime.fromisoformat(self._metadata[key].get('timestamp', '2000-01-01'))
        if datetime.now() - cached_time > timedelta(hours=max_age_hours):
            return False
        
        return self._cache_path(key).exists()
    
    def get(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        key = self._cache_key(symbol, interval)
        cache_path = self._cache_path(key)
        
        if not cache_path.exists():
            return None
        
        try:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            return df
        except Exception as e:
            logger.warning(f"Failed to read cache {key}: {e}")
            return None
    
    def put(
        self,
        symbol: str,
        interval: str,
        data: pd.DataFrame,
        source: str
    ) -> str:
        key = self._cache_key(symbol, interval)
        cache_path = self._cache_path(key)
        
        data.to_csv(cache_path)
        
        self._metadata[key] = {
            'symbol': symbol,
            'interval': interval,
            'source': source,
            'timestamp': datetime.now().isoformat(),
            'rows': len(data),
            'start_date': str(data.index.min()),
            'end_date': str(data.index.max())
        }
        self._save_metadata()
        
        return str(cache_path)
    
    def get_all_cached(self) -> Dict[str, Dict]:
        return self._metadata.copy()


class RealtimeFetcher:
    
    def __init__(self, config: Optional[FetchConfig] = None):
        self.config = config or FetchConfig()
        
        self.cache = CacheManager(self.config.cache_dir)
        self.rate_limiter = RateLimitHandler(
            requests_per_minute=self.config.requests_per_minute,
            backoff_base=self.config.backoff_base,
            max_retries=self.config.max_retries
        )
        
        self.sources = [
            YFinanceSource(),
            AlphaVantageSource(self.config.alpha_vantage_key),
        ]
        
        self.results: Dict[str, FetchResult] = {}
        self._lock = threading.Lock()
    
    def _fetch_single(
        self,
        symbol: str,
        interval: str
    ) -> FetchResult:
        start_time = time.time()
        period = self.config.periods.get(interval, 'max')
        
        if self.cache.is_cached(symbol, interval):
            cached_data = self.cache.get(symbol, interval)
            if cached_data is not None:
                logger.info(f"Cache hit: {symbol} {interval}")
                return FetchResult(
                    symbol=symbol,
                    interval=interval,
                    source='cache',
                    status='cached',
                    data=cached_data,
                    rows=len(cached_data),
                    elapsed_ms=int((time.time() - start_time) * 1000),
                    cache_path=str(self.cache._cache_path(self.cache._cache_key(symbol, interval)))
                )
        
        for source in self.sources:
            logger.info(f"Trying {source.name} for {symbol} {interval}")
            
            df, meta = source.fetch(symbol, interval, period, self.rate_limiter)
            
            if df is not None and not df.empty:
                df = self._validate_data(df)
                
                if df is not None:
                    cache_path = self.cache.put(symbol, interval, df, source.name)
                    
                    return FetchResult(
                        symbol=symbol,
                        interval=interval,
                        source=source.name,
                        status='success',
                        data=df,
                        rows=len(df),
                        elapsed_ms=int((time.time() - start_time) * 1000),
                        retries=meta.get('retries', 0),
                        rate_limited=meta.get('rate_limited', False),
                        cache_path=cache_path
                    )
            
            if meta.get('rate_limited'):
                logger.warning(f"Rate limited on {source.name} for {symbol}")
            elif meta.get('error'):
                logger.warning(f"{source.name} failed for {symbol}: {meta['error']}")
        
        return FetchResult(
            symbol=symbol,
            interval=interval,
            source='none',
            status='failed',
            elapsed_ms=int((time.time() - start_time) * 1000),
            error='All sources failed'
        )
    
    def _validate_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return None
        
        df = df[~df.index.duplicated(keep='first')]
        
        df = df.sort_index()
        
        required = ['Open', 'High', 'Low', 'Close']
        for col in required:
            if col not in df.columns:
                logger.warning(f"Missing column: {col}")
                return None
        
        df = df.dropna(subset=['Close'])
        
        total_possible = len(df)
        valid_count = df['Close'].notna().sum()
        coverage = valid_count / total_possible if total_possible > 0 else 0
        
        if coverage < self.config.min_data_coverage:
            logger.warning(f"Low data coverage: {coverage:.1%}")
        
        return df
    
    def fetch_all(self, use_cache: bool = True) -> Dict[str, FetchResult]:
        tasks = []
        for symbol in self.config.symbols:
            for interval in self.config.intervals:
                tasks.append((symbol, interval))
        
        logger.info(f"Starting parallel fetch for {len(tasks)} combinations")
        logger.info(f"Symbols: {self.config.symbols}")
        logger.info(f"Intervals: {self.config.intervals}")
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self._fetch_single, symbol, interval): (symbol, interval)
                for symbol, interval in tasks
            }
            
            for future in as_completed(futures):
                symbol, interval = futures[future]
                key = f"{symbol}_{interval}"
                
                try:
                    result = future.result(timeout=120)
                    with self._lock:
                        self.results[key] = result
                    
                    status_icon = "✓" if result.status in ['success', 'cached'] else "✗"
                    logger.info(
                        f"{status_icon} {symbol} {interval}: {result.status} "
                        f"({result.rows} rows, {result.elapsed_ms}ms, {result.source})"
                    )
                    
                except Exception as e:
                    logger.error(f"Task failed for {symbol} {interval}: {e}")
                    with self._lock:
                        self.results[key] = FetchResult(
                            symbol=symbol,
                            interval=interval,
                            source='none',
                            status='failed',
                            error=str(e)
                        )
        
        return self.results
    
    def fetch_symbol(
        self,
        symbol: str,
        intervals: Optional[List[str]] = None
    ) -> Dict[str, FetchResult]:
        intervals = intervals or self.config.intervals
        results = {}
        
        for interval in intervals:
            key = f"{symbol}_{interval}"
            results[key] = self._fetch_single(symbol, interval)
        
        return results
    
    def get_latest_data(self, symbol: str, interval: str = '1d') -> Optional[pd.DataFrame]:
        key = f"{symbol}_{interval}"
        
        if key in self.results and self.results[key].data is not None:
            return self.results[key].data
        
        result = self._fetch_single(symbol, interval)
        self.results[key] = result
        return result.data
    
    def get_summary(self) -> Dict[str, Any]:
        successful = [r for r in self.results.values() if r.status in ['success', 'cached']]
        failed = [r for r in self.results.values() if r.status == 'failed']
        rate_limited = [r for r in self.results.values() if r.rate_limited]
        
        return {
            'total_tasks': len(self.results),
            'successful': len(successful),
            'failed': len(failed),
            'rate_limited_count': len(rate_limited),
            'total_rows': sum(r.rows for r in successful),
            'sources_used': list(set(r.source for r in successful)),
            'avg_elapsed_ms': np.mean([r.elapsed_ms for r in successful]) if successful else 0,
            'rate_limiter_stats': self.rate_limiter.get_stats()
        }
    
    def save_results_log(self, path: str = "backtest_logs/fetch_log.json") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        log = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.get_summary(),
            'results': {k: v.to_dict() for k, v in self.results.items()}
        }
        
        with open(path, 'w') as f:
            json.dump(log, f, indent=2, default=str)


def fetch_all_data(config: Optional[FetchConfig] = None) -> Dict[str, FetchResult]:
    fetcher = RealtimeFetcher(config)
    return fetcher.fetch_all()


def fetch_symbol_data(
    symbol: str,
    intervals: List[str] = ['1d', '1wk']
) -> Dict[str, pd.DataFrame]:
    config = FetchConfig(symbols=[symbol], intervals=intervals)
    fetcher = RealtimeFetcher(config)
    results = fetcher.fetch_all()
    
    return {
        k: v.data for k, v in results.items()
        if v.data is not None
    }
