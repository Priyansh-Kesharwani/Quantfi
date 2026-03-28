"""DirectionalStrategy: trend/mean-reversion scoring with leveraged futures."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

from engine.crypto.engines.futures_engine import FuturesEngine, FuturesEngineConfig
from engine.crypto.models import CryptoTradeRecord, FuturesPosition, OrderIntent

logger = logging.getLogger(__name__)


@dataclass
class DirectionalConfig:
    symbol: str = "BTC/USDT:USDT"
    entry_threshold: float = 40.0
    exit_threshold: float = 15.0
    max_holding_bars: int = 168
    atr_trail_mult: float = 2.5
    kelly_fraction: float = 0.25
    max_risk_per_trade: float = 0.02
    kelly_lookback: int = 100
    min_trades_for_kelly: int = 30
    leverage: float = 3.0


class DirectionalStrategy:
    """Score-driven directional futures strategy with Kelly sizing and trailing stops."""

    def __init__(
        self,
        config: DirectionalConfig,
        engine: FuturesEngine,
    ):
        self.config = config
        self._engine = engine
        self._position: Optional[FuturesPosition] = None
        self._trade_history: List[float] = []
        self._exit_reason: str = ""

    @property
    def name(self) -> str:
        return "directional"

    @property
    def position(self) -> Optional[FuturesPosition]:
        return self._position

    def on_bar(
        self,
        bar: pd.Series,
        bar_idx: int,
        regime: str,
        score: float,
        account_equity: float,
    ) -> List[OrderIntent]:
        orders: List[OrderIntent] = []
        ts = bar.name if isinstance(bar.name, datetime) else datetime(2024, 1, 1)
        atr = bar.get("atr", 0.0)
        close = bar["close"]

        if self._position is not None:
            self._position.update_peak(close)

        if self._position is None:
            if regime != "STRESS" and abs(score) > self.config.entry_threshold:  # noqa: E501
                direction = "buy" if score > 0 else "sell"
                size_capital = self._compute_position_size(account_equity, atr, close)
                if size_capital > 0:
                    orders.append(
                        OrderIntent(
                            symbol=self.config.symbol,
                            side=direction,
                            order_type="market",
                            quantity=size_capital,
                            reason=f"score={score:.1f}, regime={regime}",
                        )
                    )
        else:
            if self._should_exit(bar, bar_idx, score, regime):
                exit_side = "sell" if self._position.direction == "long" else "buy"
                orders.append(
                    OrderIntent(
                        symbol=self.config.symbol,
                        side=exit_side,
                        order_type="market",
                        quantity=self._position.units,
                        reduce_only=True,
                        reason=self._exit_reason,
                    )
                )
        return orders

    def on_fill(self, trade: CryptoTradeRecord) -> None:
        if trade.side in ("long_entry", "short_entry"):
            pass
        elif trade.side in ("long_exit", "short_exit"):
            self._trade_history.append(trade.pnl)
            self._position = None

    def set_position(self, position: Optional[FuturesPosition]) -> None:
        self._position = position

    def force_exit(self, bar: pd.Series) -> List[OrderIntent]:
        """Force-exit the current position (used during strategy transitions)."""
        if self._position is None:
            return []
        exit_side = "sell" if self._position.direction == "long" else "buy"
        orders = [
            OrderIntent(
                symbol=self.config.symbol,
                side=exit_side,
                order_type="market",
                quantity=self._position.units,
                reduce_only=True,
                reason="strategy_transition",
            )
        ]
        return orders

    def _should_exit(self, bar: pd.Series, bar_idx: int, score: float, regime: str) -> bool:
        pos = self._position
        if pos is None:
            return False

        if abs(score) < self.config.exit_threshold:
            self._exit_reason = f"score_below_threshold={score:.1f}"
            return True

        if pos.bars_held(bar_idx) >= self.config.max_holding_bars:
            self._exit_reason = f"max_holding_bars={self.config.max_holding_bars}"
            return True

        atr = bar.get("atr", 0.0)
        atr_mult = self.config.atr_trail_mult
        if regime == "STRESS":
            atr_mult *= 0.5
        if atr > 0 and atr_mult > 0:
            if pos.direction == "long":
                trail_stop = pos.peak_price - atr_mult * atr
                if bar["close"] < trail_stop:
                    self._exit_reason = "trailing_stop_long"
                    return True
            else:
                trail_stop = pos.peak_price + atr_mult * atr
                if bar["close"] > trail_stop:
                    self._exit_reason = "trailing_stop_short"
                    return True

        return False

    def _compute_position_size(
        self, account_equity: float, atr: float, price: float
    ) -> float:
        """Kelly-fractional sizing with fallback to fixed fractional."""
        c = self.config
        if len(self._trade_history) >= c.min_trades_for_kelly:
            recent = self._trade_history[-c.kelly_lookback:]
            wins = [t for t in recent if t > 0]
            losses = [t for t in recent if t < 0]
            if wins and losses:
                p = len(wins) / len(recent)
                b = np.mean(wins) / abs(np.mean(losses))
                q = 1.0 - p
                f_star = (p * b - q) / b if b > 0 else 0.0
                f_star = max(0.0, f_star)
                fraction = min(f_star * c.kelly_fraction, c.max_risk_per_trade)
            else:
                fraction = c.max_risk_per_trade
        else:
            fraction = c.max_risk_per_trade

        return account_equity * fraction
