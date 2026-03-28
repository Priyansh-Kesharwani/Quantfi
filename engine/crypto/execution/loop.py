"""LiveExecutionLoop: main async loop for live/paper trading with risk management.

Supports two modes:
  - ``"paper"``: uses PaperExecutor with VWAP fills and latency injection
  - ``"live"``:  uses LiveCCXTExecutor (requires exchange API keys)

Candle streaming uses ccxt.pro ``watch_ohlcv`` with exponential-backoff reconnect.
State is persisted after every trade for crash recovery.

The per-bar pipeline mirrors CryptoBacktestService._run_bar_by_bar:
  1. Accumulate OHLCV history (rolling buffer)
  2. Compute scores via CryptoDirectionalScorer.compute_with_uniform_weights
  3. Compute regimes via CryptoRegimeDetector.fit_rolling
  4. Compute ATR
  5. Call strategy.on_bar(bar, bar_idx, regime, score, equity)
  6. Execute returned OrderIntents
  7. Call strategy.on_fill(trade) for each fill
"""

from __future__ import annotations

import asyncio
import logging
import os
import time as _time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
import pandas as pd

from engine.crypto.models import BotState, CryptoTradeRecord, FuturesPosition, OrderIntent
from engine.crypto.regime.detector import CryptoRegimeDetector
from engine.crypto.scoring.directional_scorer import CryptoDirectionalScorer
from engine.crypto.strategies.base import compute_first_valid_bar

logger = logging.getLogger(__name__)

_WARMUP_BARS = 600  # enough for compression_window(252) + regime_warmup(504)
_MAX_BUFFER_BARS = 1500  # cap buffer to bound O(n) scorer/regime computations


@dataclass
class RiskConfig:
    max_daily_loss_pct: float = 0.05
    max_position_notional_usd: float = 50_000.0
    max_leverage: float = 5.0
    emergency_flatten_on_disconnect: bool = True
    heartbeat_timeout_seconds: int = 30
    kill_switch_file: str = "KILL_SWITCH"
    max_retry_attempts: int = 3
    max_daily_trades: int = 100


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
        mode: Literal["paper", "live"] = "paper",
        warmup_bars: int = _WARMUP_BARS,
        futures_engine=None,
        actual_leverage: float = 3.0,
    ):
        self._strategy = strategy
        self._executor = executor
        self._adapter = adapter
        self._regime = regime_detector
        self._scorer = scorer
        self._risk = risk_config or RiskConfig()
        self._state_path = state_path
        self._mode = mode
        self._warmup_bars = warmup_bars
        self._futures_engine = futures_engine
        self._actual_leverage = actual_leverage
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0
        self._running: bool = False
        self._position: Optional[FuturesPosition] = None
        self._reconnect_delay: float = 1.0
        self._max_reconnect_delay: float = 60.0

        self._ohlcv_buffer: Optional[pd.DataFrame] = None
        self._bar_idx: int = 0
        self._first_valid: int = 0
        self._metrics = None  # set externally by BotManager
        self._cached_funding_rate: float = 0.0001
        self._funding_rate_fetched_at: Optional[datetime] = None
        self._current_trading_date: str = datetime.utcnow().strftime("%Y-%m-%d")
        self._last_candle_received_at: float = _time.monotonic()
        self._timeframe: str = ""

    async def run(self, symbol: str, timeframe: str) -> None:
        """Main loop: stream candles, compute signals, execute trades.

        Uses ccxt.pro watch_ohlcv when available, falls back to REST polling.
        """
        self._running = True
        self._timeframe = timeframe
        self._load_state()
        logger.info(
            "LiveExecutionLoop starting [mode=%s] for %s %s",
            self._mode, symbol, timeframe,
        )

        await self._warmup(symbol, timeframe)
        await self._seed_metrics_from_warmup(symbol)

        while self._running:
            try:
                await self._run_candle_loop(symbol, timeframe)
            except Exception as e:
                logger.error("Candle loop error: %s", e)
                if not self._running:
                    break
                if self._risk.emergency_flatten_on_disconnect and self._position is not None:
                    logger.warning("Disconnect with open position; flattening.")
                    await self._emergency_flatten(symbol)
                    break
                await self._backoff_reconnect()

    async def _warmup(self, symbol: str, timeframe: str) -> None:
        """Fetch historical bars to warm up scorer and regime detector."""
        effective_warmup = max(self._warmup_bars, 200)
        if effective_warmup != self._warmup_bars:
            logger.info(
                "warmup_bars=%d is too low for reliable scoring; using %d",
                self._warmup_bars, effective_warmup,
            )
        logger.info("Warming up with %d bars of history...", effective_warmup)
        try:
            ohlcv = await asyncio.to_thread(
                self._adapter.fetch_latest_n, symbol, timeframe, effective_warmup,
            )
            if ohlcv.empty or len(ohlcv) < 50:
                logger.warning("Warmup fetched only %d bars; signals may be unreliable", len(ohlcv))
                if ohlcv.empty:
                    ohlcv = pd.DataFrame(
                        columns=["open", "high", "low", "close", "volume"]
                    )
            self._ohlcv_buffer = ohlcv
            self._bar_idx = len(ohlcv)
            self._first_valid = compute_first_valid_bar(max_data=len(ohlcv))
            logger.info(
                "Warmup complete: %d bars loaded, first_valid=%d",
                len(ohlcv), self._first_valid,
            )
        except Exception as e:
            logger.error("Warmup failed: %s", e)
            self._ohlcv_buffer = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            )
            self._bar_idx = 0
            self._first_valid = 0

    async def _seed_metrics_from_warmup(self, symbol: str) -> None:
        """Process the last warmup bar to initialize metrics with real values.

        After warmup loads historical data, the dashboard would otherwise
        show $0 equity and UNKNOWN regime until the first new candle arrives.
        This runs the full scorer → regime → strategy pipeline once so the
        dashboard is immediately populated.
        """
        buf = self._ohlcv_buffer
        if buf is None or len(buf) < 2 or self._bar_idx < self._first_valid:
            if self._metrics is not None:
                balance = await self._executor.get_balance()
                self._metrics.update_position(
                    equity=balance, notional=0.0, margin=0.0,
                    unrealized=0.0, regime="WARMUP",
                )
            return

        try:
            scores = self._scorer.compute_with_uniform_weights(buf)
            regimes = self._regime.fit_rolling(buf)
        except Exception as e:
            logger.error("Warmup seed scoring failed: %s", e)
            if self._metrics is not None:
                balance = await self._executor.get_balance()
                self._metrics.update_position(
                    equity=balance, notional=0.0, margin=0.0,
                    unrealized=0.0, regime="WARMUP",
                )
            return

        prev_idx = len(buf) - 2
        signal_regime = str(regimes.iloc[prev_idx]) if prev_idx < len(regimes) else "RANGING"
        signal_score = float(scores.iloc[prev_idx]) if prev_idx < len(scores) and not np.isnan(scores.iloc[prev_idx]) else 0.0

        balance = await self._executor.get_balance()
        equity = balance
        if self._position is not None:
            close_price = float(buf.iloc[-1]["close"])
            equity += self._position.margin + self._position.unrealized_pnl(close_price)

        if self._metrics is not None:
            notional = self._position.units * float(buf.iloc[-1]["close"]) if self._position else 0.0
            margin = self._position.margin if self._position else 0.0
            unrealized = self._position.unrealized_pnl(float(buf.iloc[-1]["close"])) if self._position else 0.0
            self._metrics.update_position(
                equity=equity, notional=notional, margin=margin,
                unrealized=unrealized, regime=signal_regime,
            )

        if hasattr(self._strategy, '_current_mode') and hasattr(self._strategy, '_regime_to_mode'):
            initial_mode = self._strategy._regime_to_mode(signal_regime)
            self._strategy._current_mode = initial_mode
            logger.info("Adaptive strategy initialized to mode=%s from regime=%s", initial_mode, signal_regime)

        logger.info(
            "Warmup seed complete: equity=%.2f, regime=%s, score=%.1f, mode=%s",
            equity, signal_regime, signal_score,
            getattr(self._strategy, '_current_mode', 'N/A'),
        )

    async def _run_candle_loop(self, symbol: str, timeframe: str) -> None:
        """Inner loop: stream candles and process each one.

        Probes WebSocket (ccxt.pro watch_ohlcv) on the first iteration.
        If the exchange doesn't actually support it (regular ccxt raises
        "not supported yet"), permanently falls back to REST polling.
        """
        exchange = getattr(self._executor, "exchange", None) or getattr(
            self._executor, "_exchange", None
        )

        use_ws = self._probe_ws_support(exchange)

        if use_ws:
            logger.info("Using WebSocket (watch_ohlcv) for candle streaming")
        else:
            logger.info("Using REST polling for candle streaming")

        self._reconnect_delay = 1.0

        from engine.crypto.calendar import TIMEFRAME_TO_MS
        tf_ms = TIMEFRAME_TO_MS.get(timeframe, 3_600_000)
        poll_interval_s = min(tf_ms / 1000.0, 30.0)
        _poll_count = 0

        while self._running:
            if self._check_kill_switch():
                logger.warning("Kill switch activated. Shutting down.")
                await self._emergency_flatten(symbol)
                break

            balance = await self._executor.get_balance()
            if self._daily_pnl < -(self._risk.max_daily_loss_pct * balance):
                logger.warning("Daily loss limit hit. Flattening.")
                await self._emergency_flatten(symbol)
                break

            if use_ws:
                try:
                    candles = await exchange.watch_ohlcv(symbol, timeframe)
                    if candles:
                        latest = candles[-1]
                        self._append_candle_to_buffer(latest)
                        await self._process_bar(symbol)
                except Exception as e:
                    err_msg = str(e).lower()
                    if "not supported" in err_msg:
                        logger.warning(
                            "watch_ohlcv not supported by this exchange; "
                            "switching to REST polling permanently."
                        )
                        use_ws = False
                        continue
                    logger.error("watch_ohlcv error: %s", e)
                    raise
            else:
                poll_ok = False
                try:
                    ohlcv = await asyncio.to_thread(
                        self._adapter.fetch_latest_n, symbol, timeframe, 5,
                    )
                    if not ohlcv.empty:
                        poll_ok = True
                        new_bars = self._merge_new_bars(ohlcv)
                        if new_bars > 0:
                            logger.info("REST poll: %d new bar(s) detected", new_bars)
                        for _ in range(new_bars):
                            await self._process_bar(symbol)
                except Exception as e:
                    logger.error("REST poll error: %s", e)

                if poll_ok:
                    self._last_candle_received_at = _time.monotonic()

                _poll_count += 1
                if _poll_count % 60 == 0:
                    logger.info(
                        "Heartbeat: poll #%d, buffer=%d bars, bar_idx=%d, balance=%.2f",
                        _poll_count, len(self._ohlcv_buffer) if self._ohlcv_buffer is not None else 0,
                        self._bar_idx, await self._executor.get_balance(),
                    )

                effective_timeout = max(
                    self._risk.heartbeat_timeout_seconds,
                    tf_ms / 1000.0 * 2,
                )
                stale_s = _time.monotonic() - self._last_candle_received_at
                if stale_s > effective_timeout and self._bar_idx > 0:
                    logger.warning(
                        "No data for %.0fs (timeout=%.0fs), heartbeat timeout",
                        stale_s, effective_timeout,
                    )
                    if self._risk.emergency_flatten_on_disconnect and self._position is not None:
                        await self._emergency_flatten(symbol)
                    break

                await asyncio.sleep(poll_interval_s)

    @staticmethod
    def _probe_ws_support(exchange) -> bool:
        """Check if an exchange object genuinely supports watch_ohlcv.

        Regular ccxt exchanges have the method but it raises
        'not supported yet'. Only ccxt.pro exchanges work.
        """
        if exchange is None:
            return False
        if not hasattr(exchange, "watch_ohlcv"):
            return False
        try:
            import ccxt.pro as ccxtpro
            return isinstance(exchange, ccxtpro.Exchange)
        except (ImportError, AttributeError):
            pass
        ws_url = getattr(exchange, "urls", {}).get("ws") or getattr(
            exchange, "urls", {}
        ).get("api", {}).get("ws")
        return ws_url is not None

    def _append_candle_to_buffer(self, candle: list) -> None:
        """Append a single [ts_ms, o, h, l, c, v] candle to the OHLCV buffer."""
        ts = pd.Timestamp(candle[0], unit="ms")
        row = pd.DataFrame(
            [[candle[1], candle[2], candle[3], candle[4], candle[5]]],
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([ts]),
        )
        if self._ohlcv_buffer is None or self._ohlcv_buffer.empty:
            self._ohlcv_buffer = row
        else:
            if ts in self._ohlcv_buffer.index:
                self._ohlcv_buffer.loc[ts] = row.iloc[0]
            else:
                self._ohlcv_buffer = pd.concat([self._ohlcv_buffer, row])
                self._bar_idx += 1

    def _merge_new_bars(self, ohlcv: pd.DataFrame) -> int:
        """Merge fetched bars into buffer, return count of genuinely new bars."""
        if self._ohlcv_buffer is None or self._ohlcv_buffer.empty:
            self._ohlcv_buffer = ohlcv
            self._bar_idx = len(ohlcv)
            self._trim_buffer()
            return len(ohlcv)

        existing_end = self._ohlcv_buffer.index[-1]
        new_mask = ohlcv.index > existing_end
        new_bars = ohlcv.loc[new_mask]
        if new_bars.empty:
            return 0

        self._ohlcv_buffer = pd.concat([self._ohlcv_buffer, new_bars])
        self._ohlcv_buffer = self._ohlcv_buffer[
            ~self._ohlcv_buffer.index.duplicated(keep="last")
        ].sort_index()
        self._bar_idx += len(new_bars)
        self._trim_buffer()

        if len(new_bars) > 0 and self._ohlcv_buffer is not None and len(self._ohlcv_buffer) > 1:
            from engine.crypto.calendar import TIMEFRAME_TO_MS
            expected_gap_ms = TIMEFRAME_TO_MS.get(self._timeframe, 3_600_000) if self._timeframe else 3_600_000
            idx = self._ohlcv_buffer.index
            gaps = idx.to_series().diff().dt.total_seconds() * 1000
            large_gaps = gaps[gaps > expected_gap_ms * 1.5]
            if len(large_gaps) > 0:
                logger.warning(
                    "Detected %d candle gap(s) in buffer. Largest: %.0f ms at %s",
                    len(large_gaps),
                    large_gaps.max(),
                    large_gaps.idxmax(),
                )

        return len(new_bars)

    def _trim_buffer(self) -> None:
        """Keep buffer within _MAX_BUFFER_BARS to bound O(n) computations."""
        if self._ohlcv_buffer is not None and len(self._ohlcv_buffer) > _MAX_BUFFER_BARS:
            self._ohlcv_buffer = self._ohlcv_buffer.iloc[-_MAX_BUFFER_BARS:]

    # ------------------------------------------------------------------
    # Daily reset
    # ------------------------------------------------------------------

    def _check_daily_reset(self) -> None:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        if today != self._current_trading_date:
            logger.info(
                "Daily reset: date changed %s -> %s | pnl=%.2f trades=%d",
                self._current_trading_date, today,
                self._daily_pnl, self._daily_trades,
            )
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._current_trading_date = today
            if self._metrics is not None:
                self._metrics.reset_daily()

    # ------------------------------------------------------------------
    # Core bar processing: scorer → regime → strategy → execute → on_fill
    # ------------------------------------------------------------------

    async def _process_bar(self, symbol: str) -> None:
        """Process the latest bar through the full signal pipeline."""
        self._check_daily_reset()
        self._last_candle_received_at = _time.monotonic()

        buf = self._ohlcv_buffer
        if buf is None or len(buf) < 2:
            return

        if self._bar_idx < self._first_valid:
            logger.debug("Bar %d < first_valid %d, skipping", self._bar_idx, self._first_valid)
            return

        try:
            scores = self._scorer.compute_with_uniform_weights(buf)
            regimes = self._regime.fit_rolling(buf)
        except Exception as e:
            logger.error("Scoring/regime computation failed: %s", e)
            return

        prev_idx = len(buf) - 2
        signal_score = float(scores.iloc[prev_idx]) if prev_idx < len(scores) and not np.isnan(scores.iloc[prev_idx]) else 0.0
        signal_regime = str(regimes.iloc[prev_idx]) if prev_idx < len(regimes) else "RANGING"

        current_bar = buf.iloc[-1]
        bar_timestamp = buf.index[-1].to_pydatetime() if hasattr(buf.index[-1], "to_pydatetime") else datetime.utcnow()
        atr = self._compute_atr(buf)
        atr_val = float(atr.iloc[-2]) if len(atr) >= 2 and not np.isnan(atr.iloc[-2]) else float(current_bar["close"]) * 0.02

        bar_series = pd.Series({
            "open": current_bar["open"],
            "high": current_bar["high"],
            "low": current_bar["low"],
            "close": current_bar["close"],
            "volume": current_bar.get("volume", 0.0),
            "atr": atr_val,
        }, name=buf.index[-1])

        if self._position is not None:
            self._position.update_peak(float(current_bar["close"]))

        # --- C3: Liquidation check ---
        if self._position is not None and self._futures_engine is not None:
            liquidated, loss = self._futures_engine.check_liquidation(
                self._position,
                float(current_bar["high"]),
                float(current_bar["low"]),
            )
            if liquidated:
                logger.warning(
                    "LIQUIDATION triggered for %s %s | loss=%.2f",
                    self._position.symbol, self._position.direction, loss,
                )
                if hasattr(self._executor, "adjust_balance"):
                    self._executor.adjust_balance(-loss)
                if hasattr(self._executor, "set_position"):
                    self._executor.set_position(None)
                self._daily_pnl -= loss
                self._position = None
                self._strategy.set_position(None)
                if self._metrics is not None:
                    self._metrics.record_order_rejection()
                await self._save_state_async()

        # --- M1: Funding rate application ---
        if self._position is not None and self._futures_engine is not None:
            funding_rate = self._get_current_funding_rate(symbol)
            payment = self._futures_engine.apply_funding_if_due(
                self._position,
                funding_rate,
                float(current_bar["close"]),
                bar_timestamp,
            )
            if payment != 0.0:
                if hasattr(self._executor, "adjust_balance"):
                    self._executor.adjust_balance(-payment)
                if self._metrics is not None:
                    self._metrics.funding_paid_daily_usd += payment
                logger.info("Funding payment: %.4f USD", payment)

        balance = await self._executor.get_balance()
        equity = balance
        if self._position is not None:
            equity += self._position.margin + self._position.unrealized_pnl(float(current_bar["close"]))

        if hasattr(self._executor, "set_bar_idx"):
            self._executor.set_bar_idx(self._bar_idx)

        strategy_mode = getattr(self._strategy, '_current_mode', 'N/A')
        cooldown = getattr(self._strategy, '_cooldown_remaining', 0)
        logger.info(
            "Bar %d: close=%.2f score=%.1f regime=%s mode=%s cooldown=%d equity=%.2f",
            self._bar_idx, float(current_bar["close"]), signal_score,
            signal_regime, strategy_mode, cooldown, equity,
        )

        try:
            intents: List[OrderIntent] = self._strategy.on_bar(
                bar_series, self._bar_idx, signal_regime, signal_score, equity,
            )
        except Exception as e:
            logger.error("Strategy on_bar failed: %s", e)
            intents = []

        if intents:
            logger.info("Strategy returned %d intent(s)", len(intents))

        bar_signal_id = f"bar_{self._bar_idx}_{int(datetime.utcnow().timestamp())}"

        for i, intent in enumerate(intents):
            if intent.price is None:
                intent.price = float(current_bar["close"])
            if not intent.signal_id:
                intent.signal_id = f"{bar_signal_id}_{i}"

            if not self._risk_check(intent, balance):
                logger.warning("Risk check failed for %s", intent.reason)
                continue

            try:
                fill = await self._executor.place_order(intent)
            except Exception as e:
                logger.error("Order execution failed: %s", e)
                continue

            if fill.filled_qty > 0:
                self._strategy.on_fill(fill.trade)
                self._daily_trades += 1
                self._daily_pnl += fill.trade.pnl

                self._position = await self._executor.get_position(symbol)
                self._strategy.set_position(self._position)

                if self._metrics is not None:
                    self._metrics.record_fill(
                        expected_price=intent.price or fill.trade.price,
                        actual_price=fill.trade.price,
                        latency_ms=fill.latency_ms,
                    )
                    self._metrics.realized_pnl_daily_usd += fill.trade.pnl

                logger.info(
                    "Fill: %s %.4f @ %.2f | PnL=%.2f | Balance=%.2f",
                    fill.trade.side, fill.trade.units, fill.trade.price,
                    fill.trade.pnl, await self._executor.get_balance(),
                )
            elif self._metrics is not None:
                self._metrics.record_order_rejection()

        grid_fills = getattr(self._strategy, 'last_grid_fills', [])
        if grid_fills:
            logger.info("Grid engine produced %d fill(s) this bar", len(grid_fills))
        for gf in grid_fills:
            self._daily_trades += 1
            self._daily_pnl += gf.pnl
            if hasattr(self._executor, "adjust_balance"):
                self._executor.adjust_balance(gf.pnl)
            if hasattr(self._executor, '_ledger') and self._executor._ledger is not None:
                from engine.crypto.execution.ledger import LedgerEntry
                self._executor._ledger.append(LedgerEntry(
                    signal_id=f"grid_{gf.bar_idx}_{gf.side}",
                    timestamp=gf.timestamp.isoformat(),
                    symbol=gf.symbol,
                    side=gf.side,
                    qty_requested=gf.units,
                    qty_filled=gf.units,
                    price_requested=gf.price,
                    fill_price=gf.price,
                    slippage_bps=gf.slippage / (gf.notional + 1e-12) * 10_000,
                    latency_ms=0.0,
                    fill_status="filled",
                    source="paper",
                    reason="grid_fill",
                ))
            if self._metrics is not None:
                self._metrics.record_fill(
                    expected_price=gf.price,
                    actual_price=gf.price,
                    latency_ms=0.0,
                )
                self._metrics.realized_pnl_daily_usd += gf.pnl
            logger.info(
                "Grid fill: %s %.6f @ %.2f | PnL=%.4f",
                gf.side, gf.units, gf.price, gf.pnl,
            )

        if self._metrics is not None:
            close_price = float(buf.iloc[-1]["close"])
            notional = self._position.units * close_price if self._position else 0.0
            margin = self._position.margin if self._position else 0.0
            unrealized = self._position.unrealized_pnl(close_price) if self._position else 0.0
            self._metrics.update_position(
                equity=equity,
                notional=notional,
                margin=margin,
                unrealized=unrealized,
                regime=signal_regime,
            )

        if hasattr(self._executor, '_ledger') and self._executor._ledger is not None:
            self._executor._ledger.flush()
        await self._save_state_async()

    @staticmethod
    def _compute_atr(ohlcv: pd.DataFrame, period: int = 14) -> pd.Series:
        tail = ohlcv.iloc[-(period + 2):] if len(ohlcv) > period + 2 else ohlcv
        close = tail["close"]
        high = tail.get("high", close)
        low = tail.get("low", close)
        prev_close = close.shift(1).fillna(close)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        return tr.rolling(period, min_periods=1).mean()

    # ------------------------------------------------------------------
    # Infrastructure
    # ------------------------------------------------------------------

    async def _backoff_reconnect(self) -> None:
        logger.info("Reconnecting in %.1fs...", self._reconnect_delay)
        await asyncio.sleep(self._reconnect_delay)
        self._reconnect_delay = min(
            self._reconnect_delay * 2, self._max_reconnect_delay
        )

    def stop(self) -> None:
        self._running = False

    def _check_kill_switch(self) -> bool:
        if Path(self._risk.kill_switch_file).exists():
            return True
        if os.environ.get("CRYPTO_BOT_KILL", "").strip() == "1":
            return True
        return False

    async def _emergency_flatten(self, symbol: str) -> None:
        for attempt in range(self._risk.max_retry_attempts):
            try:
                trades = await self._executor.close_all(symbol)
                for t in trades:
                    logger.info("Emergency close: %s", t.to_dict())
                self._position = None
                self._strategy.set_position(None)
                await self._save_state_async()
                return
            except Exception as e:
                logger.error(
                    "Emergency flatten attempt %d/%d failed: %s",
                    attempt + 1, self._risk.max_retry_attempts, e,
                )
                if attempt < self._risk.max_retry_attempts - 1:
                    await asyncio.sleep(1.0 * (attempt + 1))
        logger.critical(
            "EMERGENCY FLATTEN FAILED after %d attempts. Position may be open: %s",
            self._risk.max_retry_attempts,
            self._position.to_dict() if self._position else "None",
        )

    def _save_state(self) -> None:
        state = BotState(
            position=self._position.to_dict() if self._position else None,
            daily_pnl=self._daily_pnl,
            daily_trades=self._daily_trades,
            last_bar_timestamp=datetime.utcnow().isoformat(),
            trading_date=self._current_trading_date,
        )
        tmp_path = self._state_path.with_suffix(".tmp")
        try:
            tmp_path.write_text(state.to_json())
            os.replace(str(tmp_path), str(self._state_path))
        except Exception as e:
            logger.error("State save failed: %s", e)

    async def _save_state_async(self) -> None:
        """Non-blocking state persistence via thread pool."""
        await asyncio.to_thread(self._save_state)

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            data = self._state_path.read_text()
            state = BotState.from_json(data)

            today = datetime.utcnow().strftime("%Y-%m-%d")
            saved_date = getattr(state, "trading_date", "") or ""
            if saved_date and saved_date != today:
                logger.info(
                    "State from previous day (%s); resetting daily counters",
                    saved_date,
                )
                self._daily_pnl = 0.0
                self._daily_trades = 0
            else:
                self._daily_pnl = state.daily_pnl
                self._daily_trades = state.daily_trades
            self._current_trading_date = today

            if state.position:
                try:
                    self._position = FuturesPosition.from_dict(state.position)
                    self._strategy.set_position(self._position)
                    logger.info(
                        "Restored position: %s %s",
                        self._position.symbol, self._position.direction,
                    )
                except Exception as pos_err:
                    logger.warning("Corrupted position in state file, ignoring: %s", pos_err)
                    self._position = None

            logger.info("State loaded: pnl=%.2f, trades=%d", self._daily_pnl, self._daily_trades)
        except Exception as e:
            logger.error("State load failed: %s", e)

    def _risk_check(self, intent: OrderIntent, balance: float) -> bool:
        """Pre-trade risk gate."""
        if not intent.reduce_only:
            if self._actual_leverage > self._risk.max_leverage:
                logger.warning(
                    "Leverage %.1f exceeds max %.1f",
                    self._actual_leverage, self._risk.max_leverage,
                )
                return False
            if self._daily_trades >= self._risk.max_daily_trades:
                logger.warning("Daily trade limit %d reached", self._risk.max_daily_trades)
                return False
            est_notional = intent.quantity * self._actual_leverage
            if est_notional > self._risk.max_position_notional_usd:
                logger.warning("Order exceeds max notional")
                return False
        return True

    def _get_current_funding_rate(self, symbol: str) -> float:
        """Return cached funding rate, refreshing from adapter every ~8h."""
        now = datetime.utcnow()
        if (
            self._funding_rate_fetched_at is None
            or (now - self._funding_rate_fetched_at).total_seconds() > 8 * 3600
        ):
            try:
                from datetime import timedelta
                since = now - timedelta(hours=24)
                df = self._adapter.fetch_funding_rates(symbol, since, now)
                if not df.empty:
                    self._cached_funding_rate = float(df["fundingRate"].iloc[-1])
                    logger.info("Funding rate refreshed: %.6f", self._cached_funding_rate)
            except Exception as e:
                logger.warning("Funding rate fetch failed, using cached: %s", e)
            self._funding_rate_fetched_at = now
        return self._cached_funding_rate
