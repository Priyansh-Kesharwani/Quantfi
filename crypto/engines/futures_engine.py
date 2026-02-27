"""FuturesEngine: bar-by-bar simulator for directional perpetual futures positions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

from crypto.costs import (
    compute_liquidation_price,
    execution_price,
    get_maintenance_margin,
    recalc_liquidation_price,
    trade_cost,
)
from crypto.models import CryptoTradeRecord, FuturesPosition

logger = logging.getLogger(__name__)


@dataclass
class FuturesEngineConfig:
    leverage: float = 3.0
    maker_fee_pct: float = 0.0002
    taker_fee_pct: float = 0.0005
    slippage_pct: float = 0.0002
    liquidation_fee_pct: float = 0.005
    funding_interval_hours: int = 8
    cost_preset: str = "BINANCE_FUTURES_TAKER"
    min_order_notional: float = 10.0
    slippage_multiplier: float = 1.0


class FuturesEngine:
    """Manages opening, closing, liquidation checks, and funding for futures positions."""

    def __init__(self, config: FuturesEngineConfig):
        self.config = config
        self._last_funding_hour: Optional[int] = None

    def open_position(
        self,
        symbol: str,
        direction: str,
        capital: float,
        current_price: float,
        bar_idx: int,
        timestamp: datetime,
    ) -> Optional[FuturesPosition]:
        """Open a leveraged position. Returns None if capital insufficient."""
        min_margin = self.config.min_order_notional / self.config.leverage
        if capital < min_margin * 1.1:
            return None

        notional = capital * self.config.leverage
        side = "buy" if direction == "long" else "sell"
        exec_price = execution_price(
            current_price, side, preset=self.config.cost_preset,
            slippage_multiplier=self.config.slippage_multiplier,
        )
        units = notional / exec_price
        margin = notional / self.config.leverage

        fee = trade_cost(notional, self.config.cost_preset)

        liq_price = compute_liquidation_price(
            exec_price, self.config.leverage, direction, margin - fee, units
        )

        pos = FuturesPosition(
            symbol=symbol,
            direction=direction,
            units=units,
            entry_price=exec_price,
            leverage=self.config.leverage,
            margin=margin - fee,
            liquidation_price=liq_price,
            entry_time=timestamp,
            peak_price=exec_price,
            accumulated_fees=fee,
            entry_bar_idx=bar_idx,
        )
        return pos

    def close_position(
        self,
        position: FuturesPosition,
        current_price: float,
        reason: str,
        bar_idx: int,
        timestamp: datetime,
    ) -> CryptoTradeRecord:
        """Close a position and compute realized PnL."""
        close_side = "sell" if position.direction == "long" else "buy"
        exec_price = execution_price(
            current_price, close_side, preset=self.config.cost_preset,
            slippage_multiplier=self.config.slippage_multiplier,
        )
        close_notional = exec_price * position.units
        close_fee = trade_cost(close_notional, self.config.cost_preset)

        raw_pnl = position.unrealized_pnl(exec_price)
        net_pnl = raw_pnl - close_fee - position.accumulated_funding

        trade_side = (
            "long_exit" if position.direction == "long" else "short_exit"
        )

        return CryptoTradeRecord(
            timestamp=timestamp,
            symbol=position.symbol,
            side=trade_side,
            units=position.units,
            price=exec_price,
            notional=close_notional,
            fee=position.accumulated_fees + close_fee,
            slippage=abs(exec_price - current_price) * position.units,
            funding_paid=position.accumulated_funding,
            pnl=net_pnl,
            exit_reason=reason,
            leverage=position.leverage,
            bar_idx=bar_idx,
        )

    def check_liquidation(
        self,
        position: FuturesPosition,
        bar_high: float,
        bar_low: float,
    ) -> Tuple[bool, float]:
        """Check if liquidation was triggered during this bar.

        Returns (liquidated: bool, loss_amount: float).
        """
        liq = position.liquidation_price
        if position.direction == "long":
            if bar_low <= liq:
                loss = position.margin + position.margin * self.config.liquidation_fee_pct
                return True, loss
        else:
            if bar_high >= liq:
                loss = position.margin + position.margin * self.config.liquidation_fee_pct
                return True, loss
        return False, 0.0

    def apply_funding_if_due(
        self,
        position: FuturesPosition,
        funding_rate: float,
        mark_price: float,
        bar_timestamp: datetime,
    ) -> float:
        """Apply funding if bar crosses an 8h boundary. Returns payment amount."""
        if funding_rate == 0.0:
            return 0.0

        hour = bar_timestamp.hour
        funding_hours = set(range(0, 24, self.config.funding_interval_hours))

        if hour in funding_hours and hour != self._last_funding_hour:
            self._last_funding_hour = hour
            payment = position.apply_funding(funding_rate, mark_price)
            new_liq = recalc_liquidation_price(
                position.entry_price,
                position.direction,
                position.margin,
                position.units,
                mark_price,
            )
            position.liquidation_price = new_liq
            return payment

        return 0.0

    def reset_funding_tracker(self) -> None:
        self._last_funding_hour = None
