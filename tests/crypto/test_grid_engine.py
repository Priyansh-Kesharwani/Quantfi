"""Tests for crypto.engines.grid_engine."""

from datetime import datetime

import numpy as np
import pytest

from crypto.engines.grid_engine import (
    GridConfig,
    GridEngine,
    compute_grid_levels,
    verify_grid_profitability,
)


class TestGridProfitability:
    def test_profitable_grid(self):
        config = GridConfig(
            lower_price=48_000, upper_price=52_000, num_levels=20,
            taker_fee_pct=0.0005, slippage_pct=0.0002,
        )
        assert verify_grid_profitability(config)

    def test_unprofitable_grid(self):
        config = GridConfig(
            lower_price=49_900, upper_price=50_100, num_levels=50,
            taker_fee_pct=0.0005, slippage_pct=0.0002,
        )
        assert not verify_grid_profitability(config)

    def test_degenerate_grid(self):
        config = GridConfig(lower_price=50_000, upper_price=50_000, num_levels=2)
        assert not verify_grid_profitability(config)

    def test_single_level(self):
        config = GridConfig(lower_price=48_000, upper_price=52_000, num_levels=1)
        assert not verify_grid_profitability(config)


class TestComputeLevels:
    def test_arithmetic_levels(self):
        config = GridConfig(lower_price=100, upper_price=200, num_levels=5, spacing="arithmetic")
        levels = compute_grid_levels(config)
        assert len(levels) == 5
        assert levels[0] == pytest.approx(100)
        assert levels[-1] == pytest.approx(200)
        diffs = np.diff(levels)
        assert np.allclose(diffs, diffs[0])

    def test_geometric_levels(self):
        config = GridConfig(lower_price=100, upper_price=200, num_levels=5, spacing="geometric")
        levels = compute_grid_levels(config)
        assert len(levels) == 5
        assert levels[0] == pytest.approx(100)
        assert levels[-1] == pytest.approx(200, rel=0.01)
        ratios = levels[1:] / levels[:-1]
        assert np.allclose(ratios, ratios[0])


class TestGridEngine:
    @pytest.fixture
    def grid_config(self):
        return GridConfig(
            lower_price=48_000, upper_price=52_000, num_levels=10,
            order_size_usd=100.0, taker_fee_pct=0.0005, slippage_pct=0.0001,
            max_inventory_units=1.0,
        )

    def test_initialize(self, grid_config):
        engine = GridEngine(grid_config)
        success = engine.initialize(50_000.0)
        assert success
        assert engine.is_active
        state = engine.get_state()
        assert state["n_pending"] > 0

    def test_initialize_unprofitable_fails(self):
        config = GridConfig(
            lower_price=49_990, upper_price=50_010, num_levels=50,
            taker_fee_pct=0.001, slippage_pct=0.001,
        )
        engine = GridEngine(config)
        assert not engine.initialize(50_000.0)
        assert not engine.is_active

    def test_on_bar_fills_orders(self, grid_config):
        engine = GridEngine(grid_config)
        engine.initialize(50_000.0)
        trades = engine.on_bar(50_000.0, 52_500.0, 47_500.0, 50_000.0, datetime(2024, 1, 1))
        assert len(trades) > 0

    def test_close_all(self, grid_config):
        engine = GridEngine(grid_config)
        engine.initialize(50_000.0)
        engine.on_bar(50_000.0, 50_000.0, 48_000.0, 49_000.0, datetime(2024, 1, 1))
        trades = engine.close_all(49_000.0, datetime(2024, 1, 2))
        assert not engine.is_active
        state = engine.get_state()
        assert state["inventory_units"] == pytest.approx(0.0)

    def test_inactive_engine_no_trades(self, grid_config):
        engine = GridEngine(grid_config)
        trades = engine.on_bar(50_000.0, 51_000.0, 49_000.0, 50_000.0, datetime(2024, 1, 1))
        assert len(trades) == 0

    def test_recenter(self, grid_config):
        engine = GridEngine(grid_config)
        engine.initialize(50_000.0)
        engine.recenter(55_000.0)
        state = engine.get_state()
        assert state["n_pending"] > 0
