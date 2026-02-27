from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional
import logging

import pandas as pd
import yfinance as yf

from infrastructure.adapters.symbol_resolver import SymbolResolver

logger = logging.getLogger(__name__)


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
        try:
            ticker = yf.Ticker(provider_symbol)
            if start_date and end_date:
                df = ticker.history(
                    start=start_date, end=end_date, auto_adjust=True
                )
            else:
                df = ticker.history(period=period, auto_adjust=True)
            if df is None or df.empty:
                logger.error("No data for %s", provider_symbol)
                return None
            df.columns = [
                c.title() if c[0].islower() else c for c in df.columns
            ]
            price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
            if price_cols:
                df = df.dropna(subset=price_cols, how="all")
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
