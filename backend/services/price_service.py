from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from backend.core.protocols import IPriceProvider, IFXProvider
from backend.models import PriceData

logger = logging.getLogger(__name__)


class PriceService:
    def __init__(
        self,
        price_provider: IPriceProvider,
        fx_provider: IFXProvider,
        price_repo: Any,
        asset_repo: Any,
        config: Any,
    ) -> None:
        self._price_provider = price_provider
        self._fx_provider = fx_provider
        self._price_repo = price_repo
        self._asset_repo = asset_repo
        self._config = config

    async def get_latest_price(self, symbol: str) -> dict:
        sym = symbol.upper()
        asset = await self._asset_repo.get_by_symbol(sym)
        exchange = asset.get("exchange") if asset else None
        currency = asset.get("currency", "USD") if asset else "USD"

        cached = await self._price_repo.get_latest(sym)
        if cached and isinstance(cached.get("timestamp"), str):
            cached_time = datetime.fromisoformat(cached["timestamp"])
            if (
                datetime.now(timezone.utc)
                - cached_time.replace(tzinfo=timezone.utc)
                < timedelta(minutes=self._config.cache_price_minutes)
            ):
                return cached

        price_data = self._price_provider.fetch_latest_price(sym, exchange)
        if not price_data:
            return {}

        usd_inr_rate = (
            self._fx_provider.fetch_usd_inr_rate()
            or self._config.fx_fallback_usd_inr
        )
        if currency == "INR":
            price_inr = price_data["price"]
            price_usd = price_inr / usd_inr_rate
        else:
            price_usd = price_data["price"]
            price_inr = price_data["price"] * usd_inr_rate

        price_obj = PriceData(
            symbol=sym,
            timestamp=price_data["timestamp"],
            price_usd=price_usd,
            price_inr=price_inr,
            usd_inr_rate=usd_inr_rate,
            volume=price_data.get("volume"),
        )
        doc = price_obj.model_dump()
        doc["timestamp"] = doc["timestamp"].isoformat()
        await self._price_repo.save(doc)
        return doc

    async def get_price_history(self, symbol: str, period: str = "1y") -> dict:
        sym = symbol.upper()
        df = self._price_provider.fetch_historical_data(sym, period)
        if df is None or df.empty:
            return {}
        history = []
        for date, row in df.iterrows():
            history.append({
                "date": date.isoformat(),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row["Volume"]) if "Volume" in row else None,
            })
        return {"symbol": sym, "history": history}
