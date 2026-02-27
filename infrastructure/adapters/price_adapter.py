from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import logging

import pandas as pd
import yfinance as yf

from infrastructure.adapters.symbol_resolver import SymbolResolver

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CACHE_DIRS = [_PROJECT_ROOT / "data_cache", _PROJECT_ROOT / "raw_data"]


def _cache_key(provider_symbol: str) -> str:
    return provider_symbol.replace("=", "_").replace("^", "_")


def _load_from_cache(provider_symbol: str) -> Optional[pd.DataFrame]:
    key = _cache_key(provider_symbol)
    candidates = [
        d / f"{key}_1d.csv" for d in _CACHE_DIRS
    ] + [
        d / f"{provider_symbol}.csv" for d in _CACHE_DIRS
    ] + [
        d / f"{key}.csv" for d in _CACHE_DIRS
    ]
    for path in candidates:
        if path.exists():
            try:
                df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
                if hasattr(df.index, "tz") and df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                elif df.index.dtype == "object":
                    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
                else:
                    try:
                        df.index = df.index.tz_localize(None)
                    except TypeError:
                        pass
                df.columns = [c.title() if c[0].islower() else c for c in df.columns]
                price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
                if price_cols:
                    df = df.dropna(subset=price_cols, how="all")
                if not df.empty:
                    logger.info(
                        "Loaded %d rows for %s from cache: %s",
                        len(df), provider_symbol, path,
                    )
                    return df
            except Exception as exc:
                logger.warning("Cache read failed for %s (%s): %s", provider_symbol, path, exc)
    return None


def _save_to_cache(provider_symbol: str, df: pd.DataFrame) -> None:
    cache_dir = _CACHE_DIRS[0]
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(provider_symbol)
    path = cache_dir / f"{key}_1d.csv"
    try:
        df.to_csv(path)
        logger.info("Saved %d rows for %s to cache: %s", len(df), provider_symbol, path)
    except Exception as exc:
        logger.warning("Cache save failed for %s: %s", provider_symbol, exc)


class PriceAdapter:
    def __init__(self, config: Any) -> None:
        self._config = config
        self._history_period = getattr(config, "history_period", "2y")
        self._latest_period = getattr(config, "latest_price_period", "5d")
        self._aliases = getattr(config, "symbol_aliases", None) or {}
        self._suffixes = getattr(config, "exchange_suffixes", None) or {}

    def _resolve(self, symbol: str, exchange: Optional[str] = None) -> str:
        return SymbolResolver.resolve(
            symbol, exchange, self._aliases, self._suffixes
        )

    def fetch_historical_data(
        self,
        symbol: str,
        period: str = "2y",
        exchange: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        provider_symbol = self._resolve(symbol, exchange)
        logger.info(
            "Fetching data for %s (period=%s, start=%s, end=%s)",
            provider_symbol, period, start_date, end_date,
        )
        df = self._fetch_live(provider_symbol, period, start_date, end_date)
        if df is not None and not df.empty:
            _save_to_cache(provider_symbol, df)
            return df

        logger.warning("Live fetch failed for %s, trying local cache", provider_symbol)
        df = _load_from_cache(provider_symbol)
        if df is not None and not df.empty:
            if start_date:
                ts = pd.Timestamp(start_date).tz_localize(None) if pd.Timestamp(start_date).tzinfo else pd.Timestamp(start_date)
                df = df.loc[df.index >= ts]
            if end_date:
                ts = pd.Timestamp(end_date).tz_localize(None) if pd.Timestamp(end_date).tzinfo else pd.Timestamp(end_date)
                df = df.loc[df.index <= ts]
            if not df.empty:
                return df

        logger.error("No data available for %s (live or cached)", provider_symbol)
        return None

    def _fetch_live(
        self,
        provider_symbol: str,
        period: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
    ) -> Optional[pd.DataFrame]:
        try:
            ticker = yf.Ticker(provider_symbol)
            if start_date and end_date:
                df = ticker.history(
                    start=start_date, end=end_date, auto_adjust=True
                )
            else:
                df = ticker.history(period=period, auto_adjust=True)
            if df is None or df.empty:
                return None
            df.columns = [
                c.title() if c[0].islower() else c for c in df.columns
            ]
            price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
            if price_cols:
                df = df.dropna(subset=price_cols, how="all")
            if hasattr(df.index, "tz") and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            logger.info(
                "Fetched %d rows for %s: %s -> %s",
                len(df),
                provider_symbol,
                df.index.min().date(),
                df.index.max().date(),
            )
            return df
        except Exception as e:
            logger.error("Fetch failed for %s: %s", provider_symbol, e)
            return None

    def fetch_latest_price(
        self, symbol: str, exchange: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        try:
            provider_symbol = self._resolve(symbol, exchange)
            ticker = yf.Ticker(provider_symbol)
            df = ticker.history(period=self._latest_period)
            if df is not None and not df.empty:
                latest = df.iloc[-1]
                return {
                    "price": float(latest["Close"]),
                    "volume": float(latest["Volume"]) if "Volume" in latest else None,
                    "timestamp": datetime.now(),
                    "high": float(latest["High"]),
                    "low": float(latest["Low"]),
                    "open": float(latest["Open"]),
                }
            return None
        except Exception as e:
            logger.error("Latest price error for %s: %s", symbol, e)
            return None
