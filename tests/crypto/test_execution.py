"""Tests for crypto.execution (executor, simulated, loop)."""

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from engine.crypto.engines.futures_engine import FuturesEngineConfig
from engine.crypto.execution.executor import IExecutor
from engine.crypto.execution.loop import LiveExecutionLoop, RiskConfig
from engine.crypto.execution.simulated import SimulatedExecutor
from engine.crypto.models import BotState, OrderIntent


class TestSimulatedExecutor:
    @pytest.fixture
    def executor(self):
        return SimulatedExecutor(
            FuturesEngineConfig(cost_preset="ZERO"),
            current_price_fn=lambda: 50_000.0,
        )

    def test_place_order(self, executor):
        intent = OrderIntent(
            symbol="BTC/USDT:USDT",
            side="buy",
            order_type="market",
            quantity=0.1,
        )
        result = asyncio.run(executor.place_order(intent))
        assert result.is_complete
        assert result.filled_qty == pytest.approx(0.1)
        assert result.remaining_qty == pytest.approx(0.0)

    def test_cancel_always_succeeds(self, executor):
        result = asyncio.run(executor.cancel_order("abc"))
        assert result is True

    def test_get_balance(self, executor):
        executor.set_balance(20_000.0)
        balance = asyncio.run(executor.get_balance())
        assert balance == pytest.approx(20_000.0)

    def test_protocol_compliance(self, executor):
        assert isinstance(executor, IExecutor)


class TestBotStatePersistence:
    def test_save_and_load(self):
        state = BotState(
            daily_pnl=250.0,
            daily_trades=5,
            last_bar_timestamp="2024-01-01T12:00:00",
            ic_ewma_weights=[0.1, 0.2, 0.3],
        )
        json_str = state.to_json()
        loaded = BotState.from_json(json_str)
        assert loaded.daily_pnl == pytest.approx(250.0)
        assert loaded.daily_trades == 5
        assert len(loaded.ic_ewma_weights) == 3

    def test_file_persistence(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            state = BotState(daily_pnl=100.0, daily_trades=3)
            path.write_text(state.to_json())
            loaded = BotState.from_json(path.read_text())
            assert loaded.daily_pnl == pytest.approx(100.0)
        finally:
            path.unlink(missing_ok=True)


class TestRiskConfig:
    def test_defaults(self):
        rc = RiskConfig()
        assert rc.max_daily_loss_pct == pytest.approx(0.05)
        assert rc.max_leverage == pytest.approx(5.0)
        assert rc.max_retry_attempts == 3

    def test_custom(self):
        rc = RiskConfig(max_daily_loss_pct=0.10, max_leverage=10.0)
        assert rc.max_daily_loss_pct == pytest.approx(0.10)
