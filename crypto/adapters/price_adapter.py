"""CryptoPriceAdapter: CCXT-based data fetching with Parquet cache."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from crypto.adapters.symbol_resolver import resolve_symbol
from crypto.calendar import TIMEFRAME_TO_MS

logger = logging.getLogger(__name__)


def _is_retriable(exc: BaseException) -> bool:
    try:
        import ccxt
        if isinstance(exc, (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout)):
            return True
        if isinstance(exc, (ccxt.AuthenticationError, ccxt.BadSymbol)):
            return False
    except ImportError:
        pass
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    return False


@dataclass(frozen=True)
class ExchangeConfig:
    """Configuration for a specific exchange."""

    exchange_id: str = "binance"
    max_candles_per_call: int = 1500
    rate_limit_seconds: float = 0.35
    default_type: str = "swap"
    options: Dict[str, Any] = field(default_factory=lambda: {"defaultType": "swap"})


EXCHANGE_CONFIGS = {
    "binance": ExchangeConfig("binance", 1500, 0.35, "swap", {"defaultType": "swap"}),
    "bybit": ExchangeConfig("bybit", 1000, 0.12, "swap", {"defaultType": "swap"}),
    "okx": ExchangeConfig("okx", 300, 0.15, "swap", {"defaultType": "swap"}),
}


class CryptoPriceAdapter:
    """Fetches OHLCV, funding rates, and open interest via CCXT with Parquet caching."""

    def __init__(
        self,
        exchange_config: Optional[ExchangeConfig] = None,
        cache_dir: str = "data_cache/crypto",
    ):
        self._config = exchange_config or EXCHANGE_CONFIGS["binance"]
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._exchange = None

    def _get_exchange(self) -> Any:
        if self._exchange is not None:
            return self._exchange
        try:
            import ccxt

            cls = getattr(ccxt, self._config.exchange_id)
            self._exchange = cls({"options": self._config.options, "enableRateLimit": True})
            return self._exchange
        except ImportError:
            raise RuntimeError("ccxt is required: pip install ccxt")

    def _cache_path(self, symbol: str, timeframe: str, data_type: str = "ohlcv") -> Path:
        safe_sym = symbol.replace("/", "_").replace(":", "_")
        return self._cache_dir / f"{safe_sym}_{timeframe}_{data_type}.parquet"

    def _load_cache(self, path: Path) -> Optional[pd.DataFrame]:
        if path.exists():
            try:
                df = pd.read_parquet(path)
                if not df.empty:
                    if not isinstance(df.index, pd.DatetimeIndex):
                        if "timestamp" in df.columns:
                            df = df.set_index("timestamp")
                    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
                    return df
            except Exception as e:
                logger.warning("Cache load failed for %s: %s", path, e)
        return None

    def _save_cache(self, df: pd.DataFrame, path: Path) -> None:
        try:
            df.to_parquet(path, index=True)
        except Exception as e:
            logger.warning("Cache save failed for %s: %s", path, e)

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        since: datetime,
        until: datetime,
    ) -> pd.DataFrame:
        """Fetch OHLCV data. Returns DataFrame with columns: open, high, low, close, volume.

        Index is a tz-naive DatetimeIndex (UTC). Gaps are NOT forward-filled.
        """
        cache_path = self._cache_path(symbol, timeframe)
        cached = self._load_cache(cache_path)

        since_ms = int(since.replace(tzinfo=timezone.utc).timestamp() * 1000)
        until_ms = int(until.replace(tzinfo=timezone.utc).timestamp() * 1000)

        if cached is not None and not cached.empty:
            cached_start_ms = int(cached.index[0].timestamp() * 1000)
            cached_end_ms = int(cached.index[-1].timestamp() * 1000)
            if cached_start_ms <= since_ms and cached_end_ms >= until_ms:
                mask = (cached.index >= pd.Timestamp(since)) & (
                    cached.index <= pd.Timestamp(until)
                )
                return cached.loc[mask]

        df = self._fetch_paginated(symbol, timeframe, since_ms, until_ms)

        if cached is not None and not cached.empty and not df.empty:
            df = pd.concat([cached, df])
            df = df[~df.index.duplicated(keep="last")]
            df = df.sort_index()

        if not df.empty:
            self._save_cache(df, cache_path)

        mask = (df.index >= pd.Timestamp(since)) & (df.index <= pd.Timestamp(until))
        return df.loc[mask]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception(_is_retriable),
        reraise=True,
    )
    def _fetch_single_page(self, exchange, symbol, timeframe, since, limit):
        return exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)

    def _fetch_paginated(
        self, symbol: str, timeframe: str, since_ms: int, until_ms: int
    ) -> pd.DataFrame:
        exchange = self._get_exchange()
        tf_ms = TIMEFRAME_TO_MS.get(timeframe, 3_600_000)
        all_candles = []
        current_since = since_ms

        while current_since < until_ms:
            try:
                candles = self._fetch_single_page(
                    exchange, symbol, timeframe,
                    current_since, self._config.max_candles_per_call,
                )
            except Exception as e:
                logger.error("CCXT fetch_ohlcv error after retries: %s", e)
                break

            if not candles:
                break

            all_candles.extend(candles)
            last_ts = candles[-1][0]

            if last_ts >= until_ms:
                break

            current_since = last_ts + tf_ms
            time.sleep(self._config.rate_limit_seconds)

        if not all_candles:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        return self._raw_to_df(all_candles)

    @staticmethod
    def _raw_to_df(candles: list) -> pd.DataFrame:
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def fetch_funding_rates(
        self,
        symbol: str,
        since: datetime,
        until: datetime,
    ) -> pd.DataFrame:
        """Fetch funding rate history. Returns DataFrame with columns: timestamp, fundingRate."""
        cache_path = self._cache_path(symbol, "8h", "funding")
        cached = self._load_cache(cache_path)

        since_ms = int(since.replace(tzinfo=timezone.utc).timestamp() * 1000)
        until_ms = int(until.replace(tzinfo=timezone.utc).timestamp() * 1000)

        try:
            exchange = self._get_exchange()
            if hasattr(exchange, "fetch_funding_rate_history"):
                raw = exchange.fetch_funding_rate_history(symbol, since=since_ms, limit=1000)
            else:
                logger.warning("Exchange does not support fetch_funding_rate_history")
                return pd.DataFrame(columns=["timestamp", "fundingRate"])
        except Exception as e:
            logger.error("Funding rate fetch error: %s", e)
            if cached is not None:
                return cached
            return pd.DataFrame(columns=["timestamp", "fundingRate"])

        if not raw:
            if cached is not None:
                return cached
            return pd.DataFrame(columns=["timestamp", "fundingRate"])

        records = []
        for entry in raw:
            ts = entry.get("timestamp") or entry.get("datetime")
            rate = entry.get("fundingRate", 0.0)
            if ts is not None and rate is not None:
                records.append({"timestamp": pd.Timestamp(ts, unit="ms" if isinstance(ts, (int, float)) else None), "fundingRate": float(rate)})

        df = pd.DataFrame(records)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
            df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
            if cached is not None and not cached.empty:
                df = pd.concat([cached, df])
                df = df[~df.index.duplicated(keep="last")]
                df = df.sort_index()
            self._save_cache(df, cache_path)

        return df

    def fetch_open_interest(
        self,
        symbol: str,
        timeframe: str,
        since: datetime,
        until: datetime,
    ) -> pd.DataFrame:
        """Fetch open interest data."""
        try:
            exchange = self._get_exchange()
            since_ms = int(since.replace(tzinfo=timezone.utc).timestamp() * 1000)
            if hasattr(exchange, "fetch_open_interest_history"):
                raw = exchange.fetch_open_interest_history(
                    symbol, timeframe=timeframe, since=since_ms, limit=500
                )
                if raw:
                    records = [
                        {"timestamp": pd.Timestamp(r["timestamp"], unit="ms"), "openInterest": r.get("openInterestAmount", r.get("openInterest", 0))}
                        for r in raw
                        if r.get("timestamp")
                    ]
                    df = pd.DataFrame(records)
                    if not df.empty:
                        df = df.set_index("timestamp")
                        df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
                        return df
        except Exception as e:
            logger.warning("Open interest fetch failed: %s", e)

        return pd.DataFrame(columns=["timestamp", "openInterest"])
