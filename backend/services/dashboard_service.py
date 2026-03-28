from __future__ import annotations

from typing import Any

from backend.repositories.asset_repository import AssetRepository
from backend.repositories.score_repository import ScoreRepository
from backend.repositories.price_repository import PriceRepository
from backend.repositories.indicator_repository import IndicatorRepository


class DashboardService:
    def __init__(
        self,
        asset_repo: AssetRepository,
        score_repo: ScoreRepository,
        price_repo: PriceRepository,
        indicator_repo: IndicatorRepository,
    ) -> None:
        self._assets = asset_repo
        self._scores = score_repo
        self._prices = price_repo
        self._indicators = indicator_repo

    async def get_snapshot(self) -> dict:
        assets = await self._assets.get_all_active()
        result = []
        for asset in assets:
            symbol = asset["symbol"]
            score = await self._scores.get_latest(symbol)
            price = await self._prices.get_latest(symbol)
            indicators = await self._indicators.get_latest(symbol)
            result.append({
                "asset": asset,
                "score": score,
                "latest_price": price,
                "indicators": indicators,
            })
        return {"assets": result}
