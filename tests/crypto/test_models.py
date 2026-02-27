"""Tests for crypto.models."""

import json
from datetime import datetime

import pytest

from crypto.models import (
    BotState,
    CryptoTradeRecord,
    FillResult,
    FuturesPosition,
    GridOrder,
    OrderIntent,
)


@pytest.fixture
def long_position():
    return FuturesPosition(
        symbol="BTC/USDT:USDT",
        direction="long",
        units=0.1,
        entry_price=50_000.0,
        leverage=3.0,
        margin=50_000 * 0.1 / 3.0,
        liquidation_price=33_500.0,
        entry_time=datetime(2024, 1, 1, 12, 0),
        peak_price=50_000.0,
    )


@pytest.fixture
def short_position():
    return FuturesPosition(
        symbol="ETH/USDT:USDT",
        direction="short",
        units=1.0,
        entry_price=3_000.0,
        leverage=5.0,
        margin=3_000.0 / 5.0,
        liquidation_price=3_500.0,
        entry_time=datetime(2024, 1, 1, 12, 0),
        peak_price=3_000.0,
    )


class TestFuturesPosition:
    def test_notional(self, long_position):
        assert long_position.notional == pytest.approx(5_000.0)

    def test_unrealized_pnl_long_profit(self, long_position):
        pnl = long_position.unrealized_pnl(51_000.0)
        assert pnl == pytest.approx(100.0)  # 0.1 * (51000 - 50000)

    def test_unrealized_pnl_long_loss(self, long_position):
        pnl = long_position.unrealized_pnl(49_000.0)
        assert pnl == pytest.approx(-100.0)

    def test_unrealized_pnl_short_profit(self, short_position):
        pnl = short_position.unrealized_pnl(2_900.0)
        assert pnl == pytest.approx(100.0)

    def test_unrealized_pnl_short_loss(self, short_position):
        pnl = short_position.unrealized_pnl(3_100.0)
        assert pnl == pytest.approx(-100.0)

    def test_current_equity(self, long_position):
        equity = long_position.current_equity(51_000.0)
        expected_margin = 5_000.0 / 3.0
        assert equity == pytest.approx(expected_margin + 100.0)

    def test_apply_funding_long_pays(self, long_position):
        original_margin = long_position.margin
        payment = long_position.apply_funding(0.0001, 50_000.0)
        expected_payment = 1.0 * 50_000.0 * 0.1 * 0.0001
        assert payment == pytest.approx(expected_payment)
        assert long_position.margin == pytest.approx(original_margin - expected_payment)
        assert long_position.accumulated_funding == pytest.approx(expected_payment)

    def test_apply_funding_short_receives(self, short_position):
        original_margin = short_position.margin
        payment = short_position.apply_funding(0.0001, 3_000.0)
        expected_payment = -1.0 * 3_000.0 * 1.0 * 0.0001
        assert payment == pytest.approx(expected_payment)
        assert short_position.margin == pytest.approx(original_margin - expected_payment)

    def test_bars_held(self, long_position):
        assert long_position.bars_held(10) == 10

    def test_update_peak_long(self, long_position):
        long_position.update_peak(52_000.0)
        assert long_position.peak_price == 52_000.0
        long_position.update_peak(51_000.0)
        assert long_position.peak_price == 52_000.0

    def test_update_peak_short(self, short_position):
        short_position.update_peak(2_800.0)
        assert short_position.peak_price == 2_800.0
        short_position.update_peak(2_900.0)
        assert short_position.peak_price == 2_800.0

    def test_serialization_roundtrip(self, long_position):
        d = long_position.to_dict()
        restored = FuturesPosition.from_dict(d)
        assert restored.symbol == long_position.symbol
        assert restored.direction == long_position.direction
        assert restored.units == pytest.approx(long_position.units)
        assert restored.entry_price == pytest.approx(long_position.entry_price)
        assert restored.margin == pytest.approx(long_position.margin)

    def test_unrealized_pnl_pct(self, long_position):
        pct = long_position.unrealized_pnl_pct(51_000.0)
        expected = 100.0 / long_position.margin
        assert pct == pytest.approx(expected)


class TestGridOrder:
    def test_creation(self):
        order = GridOrder(
            level_idx=5, price=49_500.0, side="buy", status="pending", quantity=0.01
        )
        assert order.status == "pending"
        assert order.fill_time is None

    def test_serialization(self):
        order = GridOrder(
            level_idx=3,
            price=50_500.0,
            side="sell",
            status="filled",
            quantity=0.01,
            fill_time=datetime(2024, 1, 1, 13, 0),
            fill_price=50_510.0,
        )
        d = order.to_dict()
        restored = GridOrder.from_dict(d)
        assert restored.fill_time == order.fill_time
        assert restored.fill_price == pytest.approx(order.fill_price)


class TestCryptoTradeRecord:
    def test_creation(self):
        rec = CryptoTradeRecord(
            timestamp=datetime(2024, 1, 1),
            symbol="BTC/USDT:USDT",
            side="long_entry",
            units=0.1,
            price=50_000.0,
            notional=5_000.0,
            fee=2.5,
            slippage=1.0,
            funding_paid=0.0,
            pnl=0.0,
            exit_reason="",
            leverage=3.0,
            bar_idx=0,
        )
        d = rec.to_dict()
        assert d["symbol"] == "BTC/USDT:USDT"
        assert d["side"] == "long_entry"


class TestOrderIntent:
    def test_market_order(self):
        intent = OrderIntent(
            symbol="BTC/USDT:USDT",
            side="buy",
            order_type="market",
            quantity=0.1,
            reason="score=45.2",
        )
        assert intent.price is None
        assert intent.reduce_only is False


class TestBotState:
    def test_roundtrip(self):
        state = BotState(
            regime_history=["TRENDING", "RANGING"],
            daily_pnl=150.5,
            daily_trades=3,
            last_bar_timestamp="2024-01-01T12:00:00",
            ic_ewma_weights=[0.1, 0.2, 0.3],
        )
        json_str = state.to_json()
        restored = BotState.from_json(json_str)
        assert restored.daily_pnl == pytest.approx(150.5)
        assert restored.regime_history == ["TRENDING", "RANGING"]
        assert len(restored.ic_ewma_weights) == 3
