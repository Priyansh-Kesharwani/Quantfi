"""MetricsCollector: structured metrics for live bot monitoring.

Tracks equity, margin, PnL, slippage, latency, and errors.
Exposes metrics as a JSON-serializable dict suitable for
a /metrics endpoint or Prometheus export.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List

logger = logging.getLogger(__name__)

_MAX_HISTORY = 1000


@dataclass
class MetricsCollector:
    """Accumulates real-time trading metrics for dashboards and alerts."""

    equity_usd: float = 0.0
    position_notional_usd: float = 0.0
    margin_ratio: float = 0.0
    unrealized_pnl_usd: float = 0.0
    realized_pnl_daily_usd: float = 0.0
    funding_paid_daily_usd: float = 0.0
    open_orders_count: int = 0
    current_regime: str = "UNKNOWN"
    api_errors_total: int = 0
    order_rejections_total: int = 0

    _fill_slippages_bps: Deque[float] = field(default_factory=lambda: deque(maxlen=_MAX_HISTORY))
    _exec_latencies_ms: Deque[float] = field(default_factory=lambda: deque(maxlen=_MAX_HISTORY))
    _last_update: str = ""

    def record_fill(self, expected_price: float, actual_price: float, latency_ms: float) -> None:
        if expected_price > 0:
            slippage_bps = abs(actual_price - expected_price) / expected_price * 10_000
            self._fill_slippages_bps.append(slippage_bps)
        self._exec_latencies_ms.append(latency_ms)

    def record_api_error(self) -> None:
        self.api_errors_total += 1

    def record_order_rejection(self) -> None:
        self.order_rejections_total += 1

    def update_position(
        self,
        equity: float,
        notional: float,
        margin: float,
        unrealized: float,
        regime: str,
    ) -> None:
        self.equity_usd = equity
        self.position_notional_usd = notional
        self.margin_ratio = margin / equity if equity > 0 else 0.0
        self.unrealized_pnl_usd = unrealized
        self.current_regime = regime
        self._last_update = datetime.utcnow().isoformat()

    def reset_daily(self) -> None:
        self.realized_pnl_daily_usd = 0.0
        self.funding_paid_daily_usd = 0.0

    def snapshot(self) -> Dict[str, Any]:
        """Return all metrics as a JSON-serializable dictionary."""
        slippages = list(self._fill_slippages_bps)
        latencies = list(self._exec_latencies_ms)
        return {
            "equity_usd": self.equity_usd,
            "position_notional_usd": self.position_notional_usd,
            "margin_ratio": round(self.margin_ratio, 4),
            "unrealized_pnl_usd": round(self.unrealized_pnl_usd, 2),
            "realized_pnl_daily_usd": round(self.realized_pnl_daily_usd, 2),
            "funding_paid_daily_usd": round(self.funding_paid_daily_usd, 4),
            "open_orders_count": self.open_orders_count,
            "current_regime": self.current_regime,
            "api_errors_total": self.api_errors_total,
            "order_rejections_total": self.order_rejections_total,
            "avg_fill_slippage_bps": round(sum(slippages) / len(slippages), 2) if slippages else 0.0,
            "avg_exec_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
            "n_fills_tracked": len(slippages),
            "last_update": self._last_update,
        }

    def to_json(self) -> str:
        return json.dumps(self.snapshot(), indent=2)

    def check_alerts(
        self,
        max_margin_ratio: float = 0.80,
        max_daily_loss_pct: float = 0.05,
        max_unrealized_loss_pct: float = 0.03,
    ) -> List[str]:
        """Return list of triggered alert messages."""
        alerts: List[str] = []
        if self.margin_ratio > max_margin_ratio:
            alerts.append(f"CRITICAL: margin_ratio={self.margin_ratio:.2%} > {max_margin_ratio:.0%}")
        if self.equity_usd > 0:
            daily_loss_pct = -self.realized_pnl_daily_usd / self.equity_usd
            if daily_loss_pct > max_daily_loss_pct:
                alerts.append(f"CRITICAL: daily_loss={daily_loss_pct:.2%} > {max_daily_loss_pct:.0%}")
            unr_loss_pct = -self.unrealized_pnl_usd / self.equity_usd
            if unr_loss_pct > max_unrealized_loss_pct:
                alerts.append(f"HIGH: unrealized_loss={unr_loss_pct:.2%} > {max_unrealized_loss_pct:.0%}")
        if self.order_rejections_total > 10:
            alerts.append(f"HIGH: order_rejections={self.order_rejections_total}")
        return alerts
