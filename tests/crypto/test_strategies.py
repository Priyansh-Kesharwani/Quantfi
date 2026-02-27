"""Tests for crypto.strategies (directional, grid, adaptive)."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from crypto.engines.futures_engine import FuturesEngine, FuturesEngineConfig
from crypto.engines.grid_engine import GridConfig
from crypto.models import CryptoTradeRecord
from crypto.strategies.adaptive import AdaptiveConfig, AdaptiveStrategy
from crypto.strategies.base import IStrategy, compute_first_valid_bar
from crypto.strategies.directional import DirectionalConfig, DirectionalStrategy
from crypto.strategies.grid import GridStrategy


def _make_bar(close=50_000, high=50_500, low=49_500, volume=1000, atr=500, idx=None):
    return pd.Series(
        {"open": close, "high": high, "low": low, "close": close, "volume": volume, "atr": atr},
        name=idx or datetime(2024, 1, 1),
    )


class TestComputeFirstValidBar:
    def test_default(self):
        bar = compute_first_valid_bar()
        assert bar >= 504

    def test_custom(self):
        bar = compute_first_valid_bar(regime_warmup=100, z_max_window=50, compression_window=50,
                                      ic_window=100, ic_horizon=10, atr_period=14, funding_window=50)
        assert bar == 110


class TestDirectionalStrategy:
    @pytest.fixture
    def strategy(self):
        engine = FuturesEngine(FuturesEngineConfig(leverage=3.0, cost_preset="ZERO"))
        config = DirectionalConfig(entry_threshold=30.0, exit_threshold=10.0, max_holding_bars=50)
        return DirectionalStrategy(config, engine)

    def test_no_trade_below_threshold(self, strategy):
        bar = _make_bar()
        orders = strategy.on_bar(bar, 0, "TRENDING", 20.0, 10_000.0)
        assert len(orders) == 0

    def test_long_entry_above_threshold(self, strategy):
        bar = _make_bar()
        orders = strategy.on_bar(bar, 0, "TRENDING", 50.0, 10_000.0)
        assert len(orders) == 1
        assert orders[0].side == "buy"

    def test_short_entry_below_threshold(self, strategy):
        bar = _make_bar()
        orders = strategy.on_bar(bar, 0, "TRENDING", -50.0, 10_000.0)
        assert len(orders) == 1
        assert orders[0].side == "sell"

    def test_no_entry_in_stress(self, strategy):
        bar = _make_bar()
        orders = strategy.on_bar(bar, 0, "STRESS", 80.0, 10_000.0)
        assert len(orders) == 0

    def test_exit_on_low_score(self, strategy):
        engine = strategy._engine
        pos = engine.open_position("BTC/USDT:USDT", "long", 200.0, 50_000.0, 0, datetime(2024, 1, 1))
        strategy.set_position(pos)
        bar = _make_bar()
        orders = strategy.on_bar(bar, 5, "TRENDING", 5.0, 10_000.0)
        assert len(orders) == 1
        assert orders[0].reduce_only

    def test_exit_on_max_holding(self, strategy):
        engine = strategy._engine
        pos = engine.open_position("BTC/USDT:USDT", "long", 200.0, 50_000.0, 0, datetime(2024, 1, 1))
        strategy.set_position(pos)
        bar = _make_bar()
        orders = strategy.on_bar(bar, 51, "TRENDING", 40.0, 10_000.0)
        assert len(orders) == 1

    def test_protocol_compliance(self, strategy):
        assert isinstance(strategy, IStrategy)


class TestGridStrategy:
    @pytest.fixture
    def grid_strategy(self):
        gc = GridConfig(lower_price=48_000, upper_price=52_000, num_levels=10,
                        order_size_usd=100.0, taker_fee_pct=0.0005, slippage_pct=0.0001)
        return GridStrategy(gc)

    def test_activates_on_ranging(self, grid_strategy):
        bar = _make_bar(atr=1000)
        grid_strategy.on_bar(bar, 0, "RANGING", 0.0, 10_000.0)
        assert grid_strategy.grid_engine.is_active

    def test_closes_on_trending(self, grid_strategy):
        bar = _make_bar(atr=1000)
        grid_strategy.on_bar(bar, 0, "RANGING", 0.0, 10_000.0)
        assert grid_strategy.grid_engine.is_active
        bar2 = _make_bar()
        grid_strategy.on_bar(bar2, 1, "TRENDING", 0.0, 10_000.0)
        assert not grid_strategy.grid_engine.is_active


class TestAdaptiveStrategy:
    @pytest.fixture
    def adaptive(self):
        engine = FuturesEngine(FuturesEngineConfig(leverage=3.0, cost_preset="ZERO"))
        d = DirectionalStrategy(DirectionalConfig(entry_threshold=30.0), engine)
        gc = GridConfig(lower_price=48_000, upper_price=52_000, num_levels=10,
                        order_size_usd=100.0, taker_fee_pct=0.0005, slippage_pct=0.0001)
        g = GridStrategy(gc)
        return AdaptiveStrategy(d, g, AdaptiveConfig(transition_cooldown_bars=2))

    def test_switches_to_directional_on_trending(self, adaptive):
        bar = _make_bar()
        adaptive.on_bar(bar, 0, "TRENDING", 50.0, 10_000.0)
        assert adaptive.current_mode == "directional"

    def test_switches_to_grid_on_ranging(self, adaptive):
        bar = _make_bar()
        adaptive.on_bar(bar, 0, "RANGING", 0.0, 10_000.0)
        assert adaptive.current_mode == "grid"

    def test_goes_flat_on_stress(self, adaptive):
        bar = _make_bar()
        adaptive.on_bar(bar, 0, "STRESS", 0.0, 10_000.0)
        assert adaptive.current_mode == "flat"

    def test_cooldown_prevents_immediate_trading(self, adaptive):
        bar = _make_bar()
        adaptive.on_bar(bar, 0, "TRENDING", 50.0, 10_000.0)
        orders = adaptive.on_bar(bar, 1, "RANGING", 0.0, 10_000.0)
        orders2 = adaptive.on_bar(bar, 2, "RANGING", 0.0, 10_000.0)
        assert len(orders2) == 0

    def test_protocol_compliance(self, adaptive):
        assert isinstance(adaptive, IStrategy)
