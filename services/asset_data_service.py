from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional


class AssetDataService:
    def __init__(
        self,
        price_provider: Any,
        fx_provider: Any,
        indicators: Any,
        scoring_engine: Any,
        llm_service: Any,
        config: Any,
    ) -> None:
        self._price_provider = price_provider
        self._fx_provider = fx_provider
        self._indicators = indicators
        self._scoring_engine = scoring_engine
        self._llm_service = llm_service
        self._config = config
        self._history_period = getattr(config, "history_period", "2y")
        self._fx_fallback = getattr(config, "fx_fallback_usd_inr", 83.5)

    async def fetch_and_calculate_asset_data(
        self,
        db: Any,
        symbol: str,
        exchange: Optional[str] = None,
        currency: str = "USD",
    ) -> None:
        from backend.models import (
            PriceData,
            IndicatorData,
            DCAScore,
        )

        price_data = self._price_provider.fetch_latest_price(
            symbol, exchange
        )
        if price_data:
            usd_inr_rate = None
            if currency != "INR":
                usd_inr_rate = (
                    self._fx_provider.fetch_usd_inr_rate()
                    or self._fx_fallback
                )
                price_usd = price_data["price"]
                price_inr = price_data["price"] * usd_inr_rate
            else:
                price_usd = price_inr = price_data["price"]
            if usd_inr_rate is None:
                usd_inr_rate = self._fx_fallback
            if currency == "INR" and price_usd is None:
                price_usd = price_inr / usd_inr_rate
            price_obj = PriceData(
                symbol=symbol,
                timestamp=price_data["timestamp"],
                price_usd=price_usd,
                price_inr=price_inr,
                usd_inr_rate=usd_inr_rate,
                volume=price_data.get("volume"),
            )
            doc = price_obj.model_dump()
            doc["timestamp"] = doc["timestamp"].isoformat()
            await db.price_history.insert_one(doc)

        df = self._price_provider.fetch_historical_data(
            symbol, self._history_period, exchange
        )
        if df is not None and not df.empty:
            indicators = self._indicators.calculate_all_indicators(df)
            indicator_obj = IndicatorData(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                **indicators,
            )
            doc = indicator_obj.model_dump()
            doc["timestamp"] = doc["timestamp"].isoformat()
            await db.indicators.insert_one(doc)

            current_price = float(df.iloc[-1]["Close"])
            usd_inr_rate = (
                self._fx_provider.fetch_usd_inr_rate() or self._fx_fallback
            )
            composite_score, breakdown, top_factors = (
                self._scoring_engine.calculate_composite_score(
                    indicators, current_price, usd_inr_rate
                )
            )
            zone = self._scoring_engine.get_zone(composite_score)
            explanation = await self._llm_service.generate_score_explanation(
                symbol,
                composite_score,
                breakdown.model_dump(),
                top_factors,
            )
            score_obj = DCAScore(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                composite_score=composite_score,
                zone=zone,
                breakdown=breakdown,
                explanation=explanation,
                top_factors=top_factors,
            )
            doc = score_obj.model_dump()
            doc["timestamp"] = doc["timestamp"].isoformat()
            doc["breakdown"] = breakdown.model_dump()
            await db.scores.insert_one(doc)
