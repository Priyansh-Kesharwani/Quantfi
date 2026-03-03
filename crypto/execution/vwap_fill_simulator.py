"""VWAPFillSimulator: realistic fill simulation by walking the live L2 order book.

Consumes liquidity level-by-level to compute volume-weighted average price,
slippage, and partial-fill quantities.  Falls back to the static cost model
when no order-book data is available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Tuple

from crypto.costs import execution_price

logger = logging.getLogger(__name__)


@dataclass
class VWAPResult:
    """Outcome of walking the order book for a simulated fill."""

    filled_qty: float
    vwap_price: float
    slippage_pct: float
    unfilled_qty: float
    levels_consumed: int
    best_price: float
    rejected: bool = False


@dataclass
class VWAPFillConfig:
    """Knobs for the fill simulator."""

    order_book_depth: int = 100
    max_slippage_pct: Optional[float] = 0.005  # 50 bps default tolerance
    pessimism_multiplier: float = 1.0  # >1 inflates slippage for stress tests
    cost_preset: str = "BINANCE_FUTURES_TAKER"


class VWAPFillSimulator:
    """Walks a live Level-2 order book to compute realistic fill prices."""

    def __init__(self, config: Optional[VWAPFillConfig] = None):
        self._config = config or VWAPFillConfig()

    def simulate_fill(
        self,
        exchange: Any,
        symbol: str,
        side: Literal["buy", "sell"],
        qty: float,
    ) -> VWAPResult:
        """Fetch the live order book and compute VWAP for *qty* units.

        Args:
            exchange: A ccxt exchange instance (must support fetch_order_book).
            symbol: CCXT unified symbol (e.g. ``BTC/USDT:USDT``).
            side: ``"buy"`` consumes asks, ``"sell"`` consumes bids.
            qty: Desired fill quantity in base units.

        Returns:
            VWAPResult with fill details.  If slippage exceeds tolerance the
            result is truncated (partial fill) or flagged ``rejected=True``.
        """
        try:
            ob = exchange.fetch_order_book(symbol, limit=self._config.order_book_depth)
        except Exception as e:
            logger.warning("Order-book fetch failed (%s), falling back to cost model: %s", symbol, e)
            return self._fallback_fill(symbol, side, qty)

        levels: List[Tuple[float, float]] = ob["asks"] if side == "buy" else ob["bids"]
        if not levels:
            logger.warning("Empty order book for %s %s side", symbol, side)
            return self._fallback_fill(symbol, side, qty)

        return self._walk_book(levels, side, qty)

    def simulate_fill_from_book(
        self,
        order_book: dict,
        side: Literal["buy", "sell"],
        qty: float,
    ) -> VWAPResult:
        """Same as simulate_fill but accepts a pre-fetched order book dict."""
        levels: List[Tuple[float, float]] = order_book.get("asks" if side == "buy" else "bids", [])
        if not levels:
            return VWAPResult(
                filled_qty=0.0,
                vwap_price=0.0,
                slippage_pct=0.0,
                unfilled_qty=qty,
                levels_consumed=0,
                best_price=0.0,
                rejected=True,
            )
        return self._walk_book(levels, side, qty)

    def _walk_book(
        self,
        levels: List[Tuple[float, float]],
        side: Literal["buy", "sell"],
        qty: float,
    ) -> VWAPResult:
        remaining = qty
        cost = 0.0
        filled = 0.0
        levels_consumed = 0
        best_price = levels[0][0]

        for price, level_qty in levels:
            take = min(level_qty, remaining)
            cost += take * price
            filled += take
            remaining -= take
            levels_consumed += 1
            if remaining <= 0:
                break

        if filled == 0:
            return VWAPResult(
                filled_qty=0.0,
                vwap_price=0.0,
                slippage_pct=0.0,
                unfilled_qty=qty,
                levels_consumed=0,
                best_price=best_price,
                rejected=True,
            )

        vwap = cost / filled

        if side == "buy":
            raw_slip = (vwap - best_price) / best_price
        else:
            raw_slip = (best_price - vwap) / best_price

        slippage_pct = raw_slip * self._config.pessimism_multiplier

        max_slip = self._config.max_slippage_pct
        if max_slip is not None and slippage_pct > max_slip:
            filled, cost, levels_consumed = self._truncate_to_tolerance(
                levels, side, qty, max_slip, best_price
            )
            if filled <= 0:
                return VWAPResult(
                    filled_qty=0.0,
                    vwap_price=best_price,
                    slippage_pct=slippage_pct,
                    unfilled_qty=qty,
                    levels_consumed=0,
                    best_price=best_price,
                    rejected=True,
                )
            vwap = cost / filled
            remaining = qty - filled
            if side == "buy":
                slippage_pct = (vwap - best_price) / best_price * self._config.pessimism_multiplier
            else:
                slippage_pct = (best_price - vwap) / best_price * self._config.pessimism_multiplier

        return VWAPResult(
            filled_qty=filled,
            vwap_price=vwap,
            slippage_pct=slippage_pct,
            unfilled_qty=remaining,
            levels_consumed=levels_consumed,
            best_price=best_price,
        )

    def _truncate_to_tolerance(
        self,
        levels: List[Tuple[float, float]],
        side: Literal["buy", "sell"],
        qty: float,
        max_slip: float,
        best_price: float,
    ) -> Tuple[float, float, int]:
        """Walk levels but stop before VWAP slippage exceeds max_slip."""
        remaining = qty
        cost = 0.0
        filled = 0.0
        levels_consumed = 0

        for price, level_qty in levels:
            take = min(level_qty, remaining)
            tentative_cost = cost + take * price
            tentative_filled = filled + take
            tentative_vwap = tentative_cost / tentative_filled

            if side == "buy":
                tentative_slip = (tentative_vwap - best_price) / best_price
            else:
                tentative_slip = (best_price - tentative_vwap) / best_price

            if tentative_slip * self._config.pessimism_multiplier > max_slip and filled > 0:
                break

            cost = tentative_cost
            filled = tentative_filled
            remaining -= take
            levels_consumed += 1
            if remaining <= 0:
                break

        return filled, cost, levels_consumed

    def _fallback_fill(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        qty: float,
    ) -> VWAPResult:
        """Use the static cost-model when the order book is unavailable."""
        dummy_price = 50_000.0  # caller should override via cost model
        exec_price = execution_price(
            dummy_price, side, preset=self._config.cost_preset
        )
        slip = abs(exec_price - dummy_price) / dummy_price
        return VWAPResult(
            filled_qty=qty,
            vwap_price=exec_price,
            slippage_pct=slip,
            unfilled_qty=0.0,
            levels_consumed=0,
            best_price=dummy_price,
        )
