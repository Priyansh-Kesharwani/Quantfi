"""Tests for crypto.engines.futures_engine."""

from datetime import datetime

import pytest

from engine.crypto.engines.futures_engine import FuturesEngine, FuturesEngineConfig


@pytest.fixture
def engine():
    return FuturesEngine(FuturesEngineConfig(leverage=3.0, cost_preset="ZERO"))


@pytest.fixture
def engine_with_costs():
    return FuturesEngine(FuturesEngineConfig(leverage=3.0))


class TestOpenPosition:
    def test_basic_long(self, engine):
        pos = engine.open_position(
            "BTC/USDT:USDT", "long", 1000.0, 50_000.0, 0, datetime(2024, 1, 1)
        )
        assert pos is not None
        assert pos.direction == "long"
        assert pos.leverage == 3.0
        assert pos.units > 0
        assert pos.entry_price > 0

    def test_basic_short(self, engine):
        pos = engine.open_position(
            "ETH/USDT:USDT", "short", 500.0, 3_000.0, 0, datetime(2024, 1, 1)
        )
        assert pos is not None
        assert pos.direction == "short"

    def test_insufficient_capital(self, engine):
        pos = engine.open_position(
            "BTC/USDT:USDT", "long", 0.5, 50_000.0, 0, datetime(2024, 1, 1)
        )
        assert pos is None

    def test_entry_fee_deducted(self, engine_with_costs):
        pos = engine_with_costs.open_position(
            "BTC/USDT:USDT", "long", 1000.0, 50_000.0, 0, datetime(2024, 1, 1)
        )
        assert pos is not None
        assert pos.accumulated_fees > 0
        assert pos.margin < 1000.0


class TestClosePosition:
    def test_close_long_profit(self, engine):
        pos = engine.open_position(
            "BTC/USDT:USDT", "long", 1000.0, 50_000.0, 0, datetime(2024, 1, 1)
        )
        trade = engine.close_position(pos, 51_000.0, "take_profit", 10, datetime(2024, 1, 2))
        assert trade.pnl > 0
        assert trade.side == "long_exit"
        assert trade.exit_reason == "take_profit"

    def test_close_long_loss(self, engine):
        pos = engine.open_position(
            "BTC/USDT:USDT", "long", 1000.0, 50_000.0, 0, datetime(2024, 1, 1)
        )
        trade = engine.close_position(pos, 49_000.0, "stop_loss", 5, datetime(2024, 1, 2))
        assert trade.pnl < 0

    def test_close_short_profit(self, engine):
        pos = engine.open_position(
            "BTC/USDT:USDT", "short", 1000.0, 50_000.0, 0, datetime(2024, 1, 1)
        )
        trade = engine.close_position(pos, 49_000.0, "take_profit", 10, datetime(2024, 1, 2))
        assert trade.pnl > 0
        assert trade.side == "short_exit"


class TestLiquidation:
    def test_long_not_liquidated(self, engine):
        pos = engine.open_position(
            "BTC/USDT:USDT", "long", 1000.0, 50_000.0, 0, datetime(2024, 1, 1)
        )
        liquidated, loss = engine.check_liquidation(pos, 51_000.0, 49_000.0)
        assert not liquidated

    def test_long_liquidated(self, engine):
        pos = engine.open_position(
            "BTC/USDT:USDT", "long", 1000.0, 50_000.0, 0, datetime(2024, 1, 1)
        )
        liquidated, loss = engine.check_liquidation(pos, 50_000.0, pos.liquidation_price - 100)
        assert liquidated
        assert loss > 0

    def test_short_liquidated(self, engine):
        pos = engine.open_position(
            "BTC/USDT:USDT", "short", 1000.0, 50_000.0, 0, datetime(2024, 1, 1)
        )
        liquidated, loss = engine.check_liquidation(pos, pos.liquidation_price + 100, 49_000.0)
        assert liquidated


class TestFunding:
    def test_funding_applied_at_8h(self, engine):
        pos = engine.open_position(
            "BTC/USDT:USDT", "long", 1000.0, 50_000.0, 0, datetime(2024, 1, 1, 7, 0)
        )
        original_margin = pos.margin
        payment = engine.apply_funding_if_due(
            pos, 0.0001, 50_000.0, datetime(2024, 1, 1, 8, 0)
        )
        assert payment > 0
        assert pos.margin < original_margin

    def test_no_funding_between_settlements(self, engine):
        pos = engine.open_position(
            "BTC/USDT:USDT", "long", 1000.0, 50_000.0, 0, datetime(2024, 1, 1)
        )
        payment = engine.apply_funding_if_due(
            pos, 0.0001, 50_000.0, datetime(2024, 1, 1, 3, 0)
        )
        assert payment == 0.0

    def test_funding_updates_liquidation_price(self, engine):
        pos = engine.open_position(
            "BTC/USDT:USDT", "long", 1000.0, 50_000.0, 0, datetime(2024, 1, 1, 7, 0)
        )
        original_liq = pos.liquidation_price
        engine.apply_funding_if_due(pos, 0.001, 50_000.0, datetime(2024, 1, 1, 8, 0))
        assert pos.liquidation_price > original_liq

    def test_zero_funding_rate_no_effect(self, engine):
        pos = engine.open_position(
            "BTC/USDT:USDT", "long", 1000.0, 50_000.0, 0, datetime(2024, 1, 1)
        )
        payment = engine.apply_funding_if_due(pos, 0.0, 50_000.0, datetime(2024, 1, 1, 8, 0))
        assert payment == 0.0
