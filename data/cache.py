"""
Data Caching Module

Provides disk-based caching for historical market data to avoid
repeated API calls and enable offline testing.

Author: Phase 2 Implementation
Date: 2026-02-07
"""

import os
import json
import hashlib
import pandas as pd
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class DataCache:
    """
    Disk-based cache for historical market data.
    
    Cache Structure:
    ----------------
    cache_dir/
        metadata.json          # Cache index and timestamps
        {symbol}_{interval}.parquet  # Cached data files
    
    Parameters
    ----------
    cache_dir : str
        Directory for cache storage (default: "data/cache")
    max_age_hours : int
        Maximum cache age before refresh (default: 24)
    """
    
    def __init__(
        self, 
        cache_dir: str = "data/cache",
        max_age_hours: int = 24
    ):
        self.cache_dir = Path(cache_dir)
        self.max_age = timedelta(hours=max_age_hours)
        self._ensure_cache_dir()
        self._load_metadata()
    
    def _ensure_cache_dir(self) -> None:
        """Create cache directory if it doesn't exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _metadata_path(self) -> Path:
        return self.cache_dir / "metadata.json"
    
    def _load_metadata(self) -> None:
        """Load cache metadata from disk."""
        if self._metadata_path().exists():
            try:
                with open(self._metadata_path(), "r") as f:
                    self._metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
                self._metadata = {}
        else:
            self._metadata = {}
    
    def _save_metadata(self) -> None:
        """Save cache metadata to disk."""
        try:
            with open(self._metadata_path(), "w") as f:
                json.dump(self._metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")
    
    def _cache_key(self, symbol: str, interval: str = "1d") -> str:
        """Generate cache key for a symbol/interval combination."""
        return f"{symbol.upper()}_{interval}"
    
    def _cache_path(self, key: str) -> Path:
        """Get file path for a cache key."""
        return self.cache_dir / f"{key}.parquet"
    
    def is_cached(self, symbol: str, interval: str = "1d") -> bool:
        """Check if data is cached and not expired."""
        key = self._cache_key(symbol, interval)
        
        if key not in self._metadata:
            return False
        
        cached_time = datetime.fromisoformat(self._metadata[key]["timestamp"])
        if datetime.now() - cached_time > self.max_age:
            logger.info(f"Cache expired for {key}")
            return False
        
        cache_file = self._cache_path(key)
        if not cache_file.exists():
            return False
        
        return True
    
    def get(self, symbol: str, interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Retrieve cached data for a symbol.
        
        Returns None if not cached or expired.
        """
        if not self.is_cached(symbol, interval):
            return None
        
        key = self._cache_key(symbol, interval)
        cache_file = self._cache_path(key)
        
        try:
            df = pd.read_parquet(cache_file)
            logger.info(f"Cache hit for {key}: {len(df)} rows")
            return df
        except Exception as e:
            logger.warning(f"Failed to read cache for {key}: {e}")
            return None
    
    def put(
        self, 
        symbol: str, 
        data: pd.DataFrame, 
        interval: str = "1d",
        source: str = "yfinance"
    ) -> bool:
        """
        Store data in cache.
        
        Returns True if successful.
        """
        if data is None or data.empty:
            logger.warning(f"Attempted to cache empty data for {symbol}")
            return False
        
        key = self._cache_key(symbol, interval)
        cache_file = self._cache_path(key)
        
        try:
            data.to_parquet(cache_file)
            
            self._metadata[key] = {
                "timestamp": datetime.now().isoformat(),
                "source": source,
                "rows": len(data),
                "start_date": str(data.index.min()) if hasattr(data.index, 'min') else None,
                "end_date": str(data.index.max()) if hasattr(data.index, 'max') else None
            }
            self._save_metadata()
            
            logger.info(f"Cached {key}: {len(data)} rows")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache {key}: {e}")
            return False
    
    def invalidate(self, symbol: str, interval: str = "1d") -> None:
        """Remove cached data for a symbol."""
        key = self._cache_key(symbol, interval)
        
        cache_file = self._cache_path(key)
        if cache_file.exists():
            cache_file.unlink()
        
        if key in self._metadata:
            del self._metadata[key]
            self._save_metadata()
        
        logger.info(f"Invalidated cache for {key}")
    
    def clear_all(self) -> None:
        """Clear all cached data."""
        for key in list(self._metadata.keys()):
            cache_file = self._cache_path(key)
            if cache_file.exists():
                cache_file.unlink()
        
        self._metadata = {}
        self._save_metadata()
        logger.info("Cleared all cache")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached data."""
        info = {
            "cache_dir": str(self.cache_dir),
            "max_age_hours": self.max_age.total_seconds() / 3600,
            "cached_symbols": len(self._metadata),
            "entries": {}
        }
        
        for key, meta in self._metadata.items():
            info["entries"][key] = {
                "timestamp": meta.get("timestamp"),
                "rows": meta.get("rows"),
                "source": meta.get("source"),
                "date_range": f"{meta.get('start_date')} to {meta.get('end_date')}"
            }
        
        return info
