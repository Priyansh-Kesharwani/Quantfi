"""AdaptiveStrategy: switches between Directional and Grid based on regime."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import pandas as pd

from crypto.models import CryptoTradeRecord, OrderIntent
from crypto.strategies.directional import DirectionalStrategy
from crypto.strategies.grid import GridStrategy

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveConfig:
    transition_cooldown_bars: int = 3


class AdaptiveStrategy:
    """Master strategy that delegates to Directional or Grid based on regime.

    Models transition costs (flatten old + cooldown before new entries).
    """

    def __init__(
        self,
        directional: DirectionalStrategy,
        grid: GridStrategy,
        config: Optional[AdaptiveConfig] = None,
    ):
        self._directional = directional
        self._grid = grid
        self._config = config or AdaptiveConfig()
        self._current_mode: str = "idle"
        self._cooldown_remaining: int = 0

    @property
    def name(self) -> str:
        return "adaptive"

    @property
    def current_mode(self) -> str:
        return self._current_mode

    def on_bar(
        self,
        bar: pd.Series,
        bar_idx: int,
        regime: str,
        score: float,
        account_equity: float,
    ) -> List[OrderIntent]:
        new_mode = self._regime_to_mode(regime)

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return []

        if new_mode != self._current_mode:
            transition_orders = self._execute_transition(
                self._current_mode, new_mode, bar
            )
            self._current_mode = new_mode
            self._cooldown_remaining = self._config.transition_cooldown_bars
            return transition_orders

        if self._current_mode == "directional":
            return self._directional.on_bar(bar, bar_idx, regime, score, account_equity)
        elif self._current_mode == "grid":
            return self._grid.on_bar(bar, bar_idx, regime, score, account_equity)
        else:
            return []

    def on_fill(self, trade: CryptoTradeRecord) -> None:
        if self._current_mode == "directional":
            self._directional.on_fill(trade)
        elif self._current_mode == "grid":
            self._grid.on_fill(trade)

    def _regime_to_mode(self, regime: str) -> str:
        if regime == "TRENDING":
            return "directional"
        elif regime == "RANGING":
            return "grid"
        else:
            return "flat"

    def _execute_transition(
        self,
        from_mode: str,
        to_mode: str,
        bar: pd.Series,
    ) -> List[OrderIntent]:
        """Flatten all positions when switching strategies."""
        orders: List[OrderIntent] = []

        if from_mode == "grid":
            trades = self._grid.close_all(bar)
            logger.info("Grid closed %d positions for transition", len(trades))
        elif from_mode == "directional":
            exit_orders = self._directional.force_exit(bar)
            orders.extend(exit_orders)

        return orders
