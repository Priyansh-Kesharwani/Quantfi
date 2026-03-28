"""GridEngine: bar-by-bar grid trading simulator with profitability pre-checks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np

from engine.crypto.models import CryptoTradeRecord, GridOrder

logger = logging.getLogger(__name__)


@dataclass
class GridConfig:
    lower_price: float = 0.0
    upper_price: float = 0.0
    num_levels: int = 20
    spacing: str = "geometric"
    order_size_usd: float = 100.0
    maker_fee_pct: float = 0.0002
    taker_fee_pct: float = 0.0005
    slippage_pct: float = 0.0002
    max_inventory_units: float = float("inf")
    max_grid_loss_pct: float = 0.20
    auto_recenter: bool = True
    recenter_threshold_pct: float = 0.80


def verify_grid_profitability(config: GridConfig) -> bool:
    """Check that grid step > round-trip cost. Returns False if grid guarantees losses."""
    if config.num_levels < 2 or config.upper_price <= config.lower_price:
        return False

    if config.spacing == "arithmetic":
        step_pct = (
            (config.upper_price - config.lower_price)
            / (config.num_levels - 1)
            / config.lower_price
        )
    else:
        ratio = (config.upper_price / config.lower_price) ** (
            1.0 / (config.num_levels - 1)
        )
        step_pct = ratio - 1.0

    min_step = 2 * config.taker_fee_pct + 2 * config.slippage_pct
    return step_pct > min_step


def compute_grid_levels(config: GridConfig) -> np.ndarray:
    """Compute grid price levels."""
    if config.spacing == "arithmetic":
        return np.linspace(config.lower_price, config.upper_price, config.num_levels)
    ratio = (config.upper_price / config.lower_price) ** (
        1.0 / (config.num_levels - 1)
    )
    return np.array(
        [config.lower_price * (ratio ** i) for i in range(config.num_levels)]
    )


class GridEngine:
    """Bar-by-bar grid trading engine with inventory tracking and risk controls."""

    def __init__(self, config: GridConfig):
        self.config = config
        self._levels: np.ndarray = np.array([])
        self._orders: List[GridOrder] = []
        self._inventory_units: float = 0.0
        self._inventory_cost: float = 0.0
        self._total_realized_pnl: float = 0.0
        self._total_fees: float = 0.0
        self._initial_equity: float = 0.0
        self._is_active: bool = False
        self._symbol: str = ""

    @property
    def is_active(self) -> bool:
        return self._is_active

    def initialize(
        self,
        current_price: float,
        symbol: str = "BTC/USDT:USDT",
        initial_equity: float = 0.0,
    ) -> bool:
        """Set up grid levels and place initial orders. Returns False if unprofitable."""
        if not verify_grid_profitability(self.config):
            logger.warning("Grid not profitable with current config. Skipping.")
            return False

        self._symbol = symbol
        self._levels = compute_grid_levels(self.config)
        self._orders = []
        self._inventory_units = 0.0
        self._inventory_cost = 0.0
        self._total_realized_pnl = 0.0
        self._total_fees = 0.0
        self._initial_equity = initial_equity if initial_equity > 0 else (
            self.config.order_size_usd * self.config.num_levels
        )

        for i, level in enumerate(self._levels):
            qty = self.config.order_size_usd / level
            if level < current_price:
                self._orders.append(
                    GridOrder(level_idx=i, price=level, side="buy", status="pending", quantity=qty)
                )
            elif level > current_price:
                self._orders.append(
                    GridOrder(level_idx=i, price=level, side="sell", status="pending", quantity=qty)
                )

        self._is_active = True
        return True

    def on_bar(
        self,
        bar_open: float,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        timestamp: datetime,
    ) -> List[CryptoTradeRecord]:
        """Process a single bar: check fills, manage inventory, check stop-loss."""
        if not self._is_active:
            return []

        trades = self._process_fills(bar_open, bar_high, bar_low, bar_close, timestamp)

        if self._check_grid_loss_stop():
            stop_trades = self.close_all(bar_close, timestamp, reason="grid_loss_stop")
            trades.extend(stop_trades)
            return trades

        if self.config.auto_recenter and self._should_recenter(bar_close):
            self.recenter(bar_close)

        return trades

    def _process_fills(
        self,
        bar_open: float,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        timestamp: datetime,
    ) -> List[CryptoTradeRecord]:
        """Fill orders touched during this bar, processing in price-path order."""
        trades = []
        is_bearish = bar_open > bar_close

        pending_buys = sorted(
            [o for o in self._orders if o.status == "pending" and o.side == "buy"],
            key=lambda o: o.price,
            reverse=True,
        )
        pending_sells = sorted(
            [o for o in self._orders if o.status == "pending" and o.side == "sell"],
            key=lambda o: o.price,
        )

        if is_bearish:
            for o in pending_sells:
                if bar_high >= o.price:
                    t = self._fill_order(o, timestamp)
                    if t:
                        trades.append(t)
            for o in pending_buys:
                if bar_low <= o.price:
                    t = self._fill_order(o, timestamp)
                    if t:
                        trades.append(t)
        else:
            for o in pending_buys:
                if bar_low <= o.price:
                    t = self._fill_order(o, timestamp)
                    if t:
                        trades.append(t)
            for o in pending_sells:
                if bar_high >= o.price:
                    t = self._fill_order(o, timestamp)
                    if t:
                        trades.append(t)

        return trades

    def _fill_order(self, order: GridOrder, timestamp: datetime) -> Optional[CryptoTradeRecord]:
        if order.side == "buy" and self._inventory_units >= self.config.max_inventory_units:
            return None

        slip = order.price * self.config.slippage_pct
        if order.side == "buy":
            fill_price = order.price + slip
        else:
            fill_price = order.price - slip

        notional = fill_price * order.quantity
        fee = notional * self.config.taker_fee_pct

        pnl_value = 0.0
        if order.side == "buy":
            self._inventory_units += order.quantity
            self._inventory_cost += notional + fee
        else:
            if self._inventory_units > 0:
                avg_cost_per_unit = self._inventory_cost / self._inventory_units if self._inventory_units > 0 else 0
                sell_qty = min(order.quantity, self._inventory_units)
                pnl_value = (fill_price - avg_cost_per_unit) * sell_qty - fee
                self._total_realized_pnl += pnl_value
                self._inventory_units -= sell_qty
                self._inventory_cost = max(0, self._inventory_cost - avg_cost_per_unit * sell_qty)

        self._total_fees += fee
        order.status = "filled"
        order.fill_time = timestamp
        order.fill_price = fill_price

        trade = CryptoTradeRecord(
            timestamp=timestamp,
            symbol=self._symbol,
            side="grid_buy" if order.side == "buy" else "grid_sell",
            units=order.quantity,
            price=fill_price,
            notional=notional,
            fee=fee,
            slippage=abs(fill_price - order.price) * order.quantity,
            funding_paid=0.0,
            pnl=pnl_value,
            exit_reason="grid_fill",
            leverage=1.0,
            bar_idx=0,
        )

        self._place_counter_order(order)
        return trade

    def _place_counter_order(self, filled_order: GridOrder) -> None:
        """After a fill, place the counter order at adjacent level."""
        idx = filled_order.level_idx
        if filled_order.side == "buy" and idx + 1 < len(self._levels):
            sell_price = self._levels[idx + 1]
            qty = self.config.order_size_usd / sell_price
            self._orders.append(
                GridOrder(level_idx=idx + 1, price=sell_price, side="sell", status="pending", quantity=qty)
            )
        elif filled_order.side == "sell" and idx - 1 >= 0:
            buy_price = self._levels[idx - 1]
            qty = self.config.order_size_usd / buy_price
            self._orders.append(
                GridOrder(level_idx=idx - 1, price=buy_price, side="buy", status="pending", quantity=qty)
            )

    def _check_grid_loss_stop(self) -> bool:
        if self._initial_equity <= 0:
            return False
        total_invested = sum(
            o.quantity * o.price
            for o in self._orders
            if o.status == "filled" and o.side == "buy"
        )
        if total_invested <= 0:
            return False
        loss_pct = -(self._total_realized_pnl + self._total_fees) / total_invested
        return loss_pct > self.config.max_grid_loss_pct

    def _should_recenter(self, current_price: float) -> bool:
        if len(self._levels) < 2:
            return False
        grid_range = self._levels[-1] - self._levels[0]
        center = (self._levels[0] + self._levels[-1]) / 2
        dist_from_center = abs(current_price - center)
        return dist_from_center / (grid_range / 2) > self.config.recenter_threshold_pct

    def recenter(self, new_center: float) -> None:
        """Re-center the grid around a new price."""
        half_range = (self.config.upper_price - self.config.lower_price) / 2
        self.config.lower_price = new_center - half_range
        self.config.upper_price = new_center + half_range
        self._levels = compute_grid_levels(self.config)
        old_pending = [o for o in self._orders if o.status == "pending"]
        for o in old_pending:
            o.status = "cancelled"

        for i, level in enumerate(self._levels):
            qty = self.config.order_size_usd / level
            if level < new_center:
                self._orders.append(
                    GridOrder(level_idx=i, price=level, side="buy", status="pending", quantity=qty)
                )
            elif level > new_center:
                self._orders.append(
                    GridOrder(level_idx=i, price=level, side="sell", status="pending", quantity=qty)
                )

    def close_all(
        self,
        current_price: float,
        timestamp: datetime,
        reason: str = "grid_close",
    ) -> List[CryptoTradeRecord]:
        """Flatten all inventory at market price."""
        trades = []
        if self._inventory_units > 0:
            slip = current_price * self.config.slippage_pct
            fill_price = current_price - slip
            notional = fill_price * self._inventory_units
            fee = notional * self.config.taker_fee_pct

            avg_cost = self._inventory_cost / self._inventory_units if self._inventory_units > 0 else 0
            pnl = (fill_price - avg_cost) * self._inventory_units - fee

            trades.append(
                CryptoTradeRecord(
                    timestamp=timestamp,
                    symbol=self._symbol,
                    side="grid_sell",
                    units=self._inventory_units,
                    price=fill_price,
                    notional=notional,
                    fee=fee,
                    slippage=slip * self._inventory_units,
                    funding_paid=0.0,
                    pnl=pnl,
                    exit_reason=reason,
                    leverage=1.0,
                    bar_idx=0,
                )
            )
            self._inventory_units = 0.0
            self._inventory_cost = 0.0

        for o in self._orders:
            if o.status == "pending":
                o.status = "cancelled"

        self._is_active = False
        return trades

    def get_state(self) -> dict:
        return {
            "is_active": self._is_active,
            "inventory_units": self._inventory_units,
            "inventory_cost": self._inventory_cost,
            "total_realized_pnl": self._total_realized_pnl,
            "total_fees": self._total_fees,
            "n_orders": len(self._orders),
            "n_pending": sum(1 for o in self._orders if o.status == "pending"),
        }
