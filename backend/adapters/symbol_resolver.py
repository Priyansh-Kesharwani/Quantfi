from __future__ import annotations

import time
from typing import Dict, Optional, List, Tuple
import logging

import yfinance as yf

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 3600


class SymbolResolver:
    _info_cache: Dict[str, Tuple[Dict, float]] = {}
    _MAX_CACHE_SIZE = 500

    @classmethod
    def resolve(
        cls,
        symbol: str,
        exchange: Optional[str] = None,
        aliases: Optional[Dict[str, str]] = None,
        exchange_suffixes: Optional[Dict[str, str]] = None,
    ) -> str:
        aliases = aliases or {}
        exchange_suffixes = exchange_suffixes or {}
        upper = symbol.upper()
        if upper in aliases:
            return aliases[upper]
        if exchange and exchange in exchange_suffixes:
            sfx = exchange_suffixes[exchange]
            return symbol if symbol.endswith(sfx) else f"{symbol}{sfx}"
        return symbol

    @classmethod
    def get_info(cls, symbol: str) -> Dict:
        now = time.monotonic()
        if symbol in cls._info_cache:
            info, ts = cls._info_cache[symbol]
            if now - ts < _CACHE_TTL_SECONDS:
                return info

        if len(cls._info_cache) >= cls._MAX_CACHE_SIZE:
            oldest_key = min(cls._info_cache, key=lambda k: cls._info_cache[k][1])
            del cls._info_cache[oldest_key]

        try:
            info = yf.Ticker(symbol).info or {}
            cls._info_cache[symbol] = (info, now)
            return info
        except Exception:
            return {}

    @classmethod
    def get_name(cls, symbol: str) -> str:
        info = cls.get_info(symbol)
        return info.get("longName") or info.get("shortName") or symbol

    @classmethod
    def get_news_keywords(cls, symbol: str) -> List[str]:
        info = cls.get_info(symbol)
        kw = []
        name = info.get("longName") or info.get("shortName") or ""
        if name:
            kw.append(name)
        kw.append(
            symbol.replace("=F", "")
            .replace("-USD", "")
            .replace(".NS", "")
            .replace(".BO", "")
        )
        sector = info.get("sector", "")
        if sector:
            kw.append(sector)
        qt = info.get("quoteType", "")
        if qt == "FUTURE":
            kw.append("futures commodity")
        elif qt == "CRYPTOCURRENCY":
            kw.append("crypto")
        elif qt == "ETF":
            kw.append("ETF fund")
        return [k for k in kw if k]
