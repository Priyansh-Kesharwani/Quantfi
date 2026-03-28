"""GridStrategy: range-bound grid trading for RANGING regimes."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

import pandas as pd

from engine.crypto.engines.grid_engine import GridConfig, GridEngine
from engine.crypto.models import CryptoTradeRecord, OrderIntent

logger = logging.getLogger(__name__)


class GridStrategy:
    """Activates a grid when regime is RANGING, closes when regime switches."""

    def __init__(
        self,
        grid_config: GridConfig,
        symbol: str = "BTC/USDT:USDT",
        atr_multiplier: float = 3.0,
    ):
        self._grid_config = grid_config
        self._engine = GridEngine(grid_config)
        self._symbol = symbol
        self._atr_multiplier = atr_multiplier
        self._last_grid_fills: List[CryptoTradeRecord] = []

    @property
    def name(self) -> str:
        return "grid"

    @property
    def grid_engine(self) -> GridEngine:
        return self._engine

    @property
    def last_grid_fills(self) -> List[CryptoTradeRecord]:
        """Grid fills from the most recent on_bar call (consumed by the loop)."""
        return self._last_grid_fills

    def on_bar(
        self,
        bar: pd.Series,
        bar_idx: int,
        regime: str,
        score: float,
        account_equity: float,
    ) -> List[OrderIntent]:
        self._last_grid_fills = []
        ts = bar.name if isinstance(bar.name, datetime) else datetime(2024, 1, 1)
        close = bar["close"]

        if regime != "RANGING" and self._engine.is_active:
            trades = self._engine.close_all(close, ts, reason="regime_exit")
            self._last_grid_fills = trades
            return []

        if regime == "RANGING" and not self._engine.is_active:
            atr = bar.get("atr", close * 0.02)
            half_range = atr * self._atr_multiplier
            self._grid_config.lower_price = close - half_range
            self._grid_config.upper_price = close + half_range
            success = self._engine.initialize(close, self._symbol, initial_equity=account_equity)
            if not success:
                return []

        if self._engine.is_active:
            trades = self._engine.on_bar(
                bar.get("open", close),
                bar.get("high", close),
                bar.get("low", close),
                close,
                ts,
            )
            self._last_grid_fills = trades

        return []

    def on_fill(self, trade: CryptoTradeRecord) -> None:
        pass

    def close_all(self, bar: pd.Series) -> List[CryptoTradeRecord]:
        ts = bar.name if isinstance(bar.name, datetime) else datetime(2024, 1, 1)
        return self._engine.close_all(bar["close"], ts, reason="strategy_transition")
