from __future__ import annotations

from typing import Any, Dict, Optional
from datetime import datetime, timezone

import pandas as pd

from domain.protocols import (
    IPriceProvider,
    IFXProvider,
    IIndicators,
    IScoringEngine,
)


class ScoringFacade:
    def __init__(
        self,
        indicators: IIndicators,
        scoring_engine: IScoringEngine,
        price_provider: IPriceProvider,
        fx_provider: IFXProvider,
        config: Any,
    ) -> None:
        self._indicators = indicators
        self._scoring_engine = scoring_engine
        self._price_provider = price_provider
        self._fx_provider = fx_provider
        self._config = config

    def get_indicators_for_symbol(
        self,
        symbol: str,
        period: Optional[str] = None,
        exchange: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        period = period or getattr(
            self._config, "history_period", "2y"
        )
        df = self._price_provider.fetch_historical_data(
            symbol.upper(), period=period, exchange=exchange
        )
        if df is None or df.empty:
            return None
        return self._indicators.calculate_all_indicators(df)

    def get_score_for_symbol(
        self,
        symbol: str,
        exchange: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> Optional[tuple]:
        df = self._price_provider.fetch_historical_data(
            symbol.upper(),
            period=getattr(self._config, "history_period", "2y"),
            exchange=exchange,
        )
        if df is None or df.empty:
            return None
        indicators = self._indicators.calculate_all_indicators(df)
        current_price = float(df.iloc[-1]["Close"])
        usd_inr_rate = (
            self._fx_provider.fetch_usd_inr_rate()
            or getattr(self._config, "fx_fallback_usd_inr", 83.5)
        )
        return self._scoring_engine.calculate_composite_score(
            indicators, current_price, usd_inr_rate, weights
        )
