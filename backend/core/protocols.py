from __future__ import annotations

from typing import Protocol, Optional, Dict, List, Any, runtime_checkable
from datetime import datetime

import pandas as pd


@runtime_checkable
class IPriceProvider(Protocol):
    def fetch_historical_data(
        self,
        symbol: str,
        period: str = "2y",
        exchange: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]: ...

    def fetch_latest_price(
        self, symbol: str, exchange: Optional[str] = None
    ) -> Optional[Dict[str, Any]]: ...


@runtime_checkable
class IFXProvider(Protocol):
    def fetch_usd_inr_rate(self) -> Optional[float]: ...


@runtime_checkable
class INewsProvider(Protocol):
    def fetch_latest_news(
        self,
        query: str = "",
        page_size: int = 20,
    ) -> List[Dict[str, Any]]: ...

    def fetch_news_for_assets(
        self, symbols: List[str], per_asset: int = 8
    ) -> List[Dict[str, Any]]: ...


@runtime_checkable
class IIndicators(Protocol):
    def calculate_all_indicators(self, df: pd.DataFrame) -> Dict[str, Any]: ...


@runtime_checkable
class IScoringEngine(Protocol):
    def calculate_composite_score(
        self,
        indicators: Dict[str, Any],
        current_price: float,
        usd_inr_rate: float,
        weights: Optional[Dict[str, float]] = None,
    ) -> tuple: ...

    def get_zone(self, composite_score: float) -> str: ...


@runtime_checkable
class IBacktestEngine(Protocol):
    def run_backtest(self, config: Any, df: pd.DataFrame) -> Any: ...


@runtime_checkable
class ILLMService(Protocol):
    async def generate_score_explanation(
        self,
        symbol: str,
        composite_score: float,
        breakdown: Dict[str, Any],
        top_factors: List[str],
    ) -> str: ...

    async def classify_news_event(
        self, title: str, description: str
    ) -> Dict[str, Any]: ...
