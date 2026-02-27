"""LiveExecutionLoop: main async loop for live trading with risk management.

WARNING — SKELETON ONLY
========================
This module is NOT production-ready.  The run() loop executes a single
iteration and breaks.  Before any live or paper deployment the following
must be implemented:

  1. WebSocket candle streaming (ccxt.pro watch_ohlcv)
  2. Full position + grid state persistence in BotState (see models.py)
  3. Reconnect with exponential backoff on API disconnect
  4. clientOrderId-based idempotent order placement
  5. Heartbeat / watchdog that flattens on stall
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from crypto.models import BotState, FuturesPosition, OrderIntent
from crypto.regime.detector import CryptoRegimeDetector
from crypto.scoring.directional_scorer import CryptoDirectionalScorer

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    max_daily_loss_pct: float = 0.05
    max_position_notional_usd: float = 50_000.0
    max_leverage: float = 5.0
    emergency_flatten_on_disconnect: bool = True
    heartbeat_timeout_seconds: int = 30
    kill_switch_file: str = "KILL_SWITCH"
    max_retry_attempts: int = 3


class LiveExecutionLoop:
    """Main async loop that coordinates strategy, execution, and risk management.

    Streams candles via CCXT WebSocket, computes signals, executes trades.
    Persists state for crash recovery.
    """

    def __init__(
        self,
        strategy,
        executor,
        adapter,
        regime_detector: CryptoRegimeDetector,
        scorer: CryptoDirectionalScorer,
        risk_config: Optional[RiskConfig] = None,
        state_path: Path = Path("crypto_state.json"),
    ):
        self._strategy = strategy
        self._executor = executor
        self._adapter = adapter
        self._regime = regime_detector
        self._scorer = scorer
        self._risk = risk_config or RiskConfig()
        self._state_path = state_path
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0
        self._running: bool = False
        self._position: Optional[FuturesPosition] = None

    _SKELETON_GUARD = True

    async def run(self, symbol: str, timeframe: str) -> None:
        """Main loop. Override in subclass or mock for testing."""
        if self._SKELETON_GUARD:
            raise NotImplementedError(
                "LiveExecutionLoop is a skeleton.  Set _SKELETON_GUARD = False "
                "only after implementing candle streaming, state persistence, "
                "and reconnect logic.  See module docstring."
            )
        self._running = True
        self._load_state()

        logger.info("LiveExecutionLoop starting for %s %s", symbol, timeframe)

        while self._running:
            if self._check_kill_switch():
                logger.warning("Kill switch activated. Shutting down.")
                await self._emergency_flatten(symbol)
                break

            if self._daily_pnl < -(self._risk.max_daily_loss_pct * await self._executor.get_balance()):
                logger.warning("Daily loss limit hit. Flattening.")
                await self._emergency_flatten(symbol)
                break

            break

    def stop(self) -> None:
        self._running = False

    def _check_kill_switch(self) -> bool:
        if Path(self._risk.kill_switch_file).exists():
            return True
        if os.environ.get("CRYPTO_BOT_KILL", "").strip() == "1":
            return True
        return False

    async def _emergency_flatten(self, symbol: str) -> None:
        try:
            trades = await self._executor.close_all(symbol)
            for t in trades:
                logger.info("Emergency close: %s", t.to_dict())
            self._save_state()
        except Exception as e:
            logger.error("Emergency flatten failed: %s", e)

    def _save_state(self) -> None:
        state = BotState(
            position=self._position.to_dict() if self._position else None,
            daily_pnl=self._daily_pnl,
            daily_trades=self._daily_trades,
            last_bar_timestamp=datetime.utcnow().isoformat(),
        )
        try:
            self._state_path.write_text(state.to_json())
        except Exception as e:
            logger.error("State save failed: %s", e)

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            data = self._state_path.read_text()
            state = BotState.from_json(data)
            self._daily_pnl = state.daily_pnl
            self._daily_trades = state.daily_trades
            if state.position:
                self._position = FuturesPosition.from_dict(state.position)
                logger.info("Restored position: %s %s", self._position.symbol, self._position.direction)
            logger.info("State loaded: pnl=%.2f, trades=%d", state.daily_pnl, state.daily_trades)
        except Exception as e:
            logger.error("State load failed: %s", e)

    def _risk_check(self, intent: OrderIntent, balance: float) -> bool:
        """Pre-trade risk gate."""
        if intent.quantity * (intent.price or 50_000) > self._risk.max_position_notional_usd:
            logger.warning("Order exceeds max notional")
            return False
        return True
