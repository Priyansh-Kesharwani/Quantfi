"""Domain models for the crypto trading bot."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import List, Literal, Optional


@dataclass
class FuturesPosition:
    """Isolated-margin perpetual futures position with dynamic liquidation tracking."""

    symbol: str
    direction: Literal["long", "short"]
    units: float
    entry_price: float
    leverage: float
    margin: float
    liquidation_price: float
    entry_time: datetime
    peak_price: float
    accumulated_funding: float = 0.0
    accumulated_fees: float = 0.0
    entry_bar_idx: int = 0

    @property
    def notional(self) -> float:
        return self.units * self.entry_price

    def unrealized_pnl(self, mark_price: float) -> float:
        sign = 1.0 if self.direction == "long" else -1.0
        return sign * self.units * (mark_price - self.entry_price)

    def current_equity(self, mark_price: float) -> float:
        return self.margin + self.unrealized_pnl(mark_price)

    def unrealized_pnl_pct(self, mark_price: float) -> float:
        if self.margin <= 0:
            return 0.0
        return self.unrealized_pnl(mark_price) / self.margin

    def apply_funding(self, funding_rate: float, mark_price: float) -> float:
        """Apply funding on CURRENT notional (mark_price * units).

        Longs pay when rate > 0, shorts receive.
        Returns the payment amount (positive = cost to holder).
        """
        current_notional = mark_price * self.units
        sign = 1.0 if self.direction == "long" else -1.0
        payment = sign * current_notional * funding_rate
        self.accumulated_funding += payment
        self.margin -= payment
        return payment

    def bars_held(self, current_bar_idx: int) -> int:
        return current_bar_idx - self.entry_bar_idx

    def update_peak(self, mark_price: float) -> None:
        if self.direction == "long":
            self.peak_price = max(self.peak_price, mark_price)
        else:
            self.peak_price = min(self.peak_price, mark_price)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["entry_time"] = self.entry_time.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FuturesPosition":
        d = dict(d)
        d["entry_time"] = datetime.fromisoformat(d["entry_time"])
        return cls(**d)


@dataclass
class GridOrder:
    """A single resting order within a grid."""

    level_idx: int
    price: float
    side: Literal["buy", "sell"]
    status: Literal["pending", "filled", "cancelled"]
    quantity: float
    fill_time: Optional[datetime] = None
    fill_price: Optional[float] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.fill_time is not None:
            d["fill_time"] = self.fill_time.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "GridOrder":
        d = dict(d)
        if d.get("fill_time") is not None:
            d["fill_time"] = datetime.fromisoformat(d["fill_time"])
        return cls(**d)


@dataclass
class CryptoTradeRecord:
    """Immutable record of a completed trade."""

    timestamp: datetime
    symbol: str
    side: Literal[
        "long_entry", "long_exit", "short_entry", "short_exit", "grid_buy", "grid_sell"
    ]
    units: float
    price: float
    notional: float
    fee: float
    slippage: float
    funding_paid: float
    pnl: float
    exit_reason: str
    leverage: float
    bar_idx: int

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


@dataclass
class OrderIntent:
    """An intent to place an order -- strategy output before execution."""

    symbol: str
    side: Literal["buy", "sell"]
    order_type: Literal["market", "limit"]
    quantity: float
    price: Optional[float] = None
    reduce_only: bool = False
    reason: str = ""


@dataclass
class FillResult:
    """Result of order execution, supports partial fills."""

    trade: CryptoTradeRecord
    filled_qty: float
    remaining_qty: float
    is_complete: bool


@dataclass
class BotState:
    """Serializable bot state for crash recovery."""

    position: Optional[dict] = None
    grid_orders: List[dict] = field(default_factory=list)
    regime_history: List[str] = field(default_factory=list)
    daily_pnl: float = 0.0
    daily_trades: int = 0
    last_bar_timestamp: str = ""
    ic_ewma_weights: List[float] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    @classmethod
    def from_json(cls, data: str) -> "BotState":
        return cls(**json.loads(data))
