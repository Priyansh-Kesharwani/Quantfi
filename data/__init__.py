"""
Data Module - Phase 2

Historical data fetching, caching, and preprocessing for backtesting.

Supported Assets:
- Commodities: XAU (Gold), XAG (Silver)
- US Equities: AAPL, NFLX
- Indian Equities: NIFTY 50 constituents

Features:
- Automatic caching to disk
- Timeout handling for all network calls
- Retry logic with exponential backoff
- VWAP approximation from OHLCV
- Derived features (returns, rolling std, etc.)
"""

from .fetcher import DataFetcher, TimeoutError
from .cache import DataCache
from .preprocess import preprocess_ohlcv, compute_derived_features

__all__ = [
    'DataFetcher',
    'DataCache', 
    'TimeoutError',
    'preprocess_ohlcv',
    'compute_derived_features'
]
