"""Tests for crypto.costs."""

import pytest

from crypto.costs import (
    BINANCE_USDT_M_BRACKETS,
    CRYPTO_COST_PRESETS,
    annual_funding_drag,
    compute_liquidation_price,
    execution_price,
    get_maintenance_margin,
    recalc_liquidation_price,
    trade_cost,
)

class TestCostPresets:
    def test_all_presets_have_required_keys(self):
        for name, preset in CRYPTO_COST_PRESETS.items():
            assert "fee_pct" in preset, f"{name} missing fee_pct"
            assert "slippage_pct" in preset, f"{name} missing slippage_pct"
            assert preset["fee_pct"] >= 0
            assert preset["slippage_pct"] >= 0

    def test_binance_taker_values(self):
        p = CRYPTO_COST_PRESETS["BINANCE_FUTURES_TAKER"]
        assert p["fee_pct"] == pytest.approx(0.0005)
        assert p["slippage_pct"] == pytest.approx(0.0002)

class TestMaintenanceMargin:
    def test_first_bracket(self):
        mmr, maint = get_maintenance_margin(10_000)
        assert mmr == pytest.approx(0.004)
        assert maint == 0

    def test_second_bracket(self):
        mmr, maint = get_maintenance_margin(100_000)
        assert mmr == pytest.approx(0.005)
        assert maint == 50

    def test_boundary(self):
        mmr, maint = get_maintenance_margin(50_000)
        assert mmr == pytest.approx(0.004)

    def test_large_notional(self):
        mmr, maint = get_maintenance_margin(5_000_000)
        assert mmr == pytest.approx(0.025)
        assert maint == 16_300

class TestExecutionPrice:
    def test_buy_price_higher(self):
        ep = execution_price(50_000.0, "buy")
        assert ep > 50_000.0

    def test_sell_price_lower(self):
        ep = execution_price(50_000.0, "sell")
        assert ep < 50_000.0

    def test_zero_preset(self):
        ep = execution_price(50_000.0, "buy", preset="ZERO")
        assert ep == pytest.approx(50_000.0)

    def test_large_order_has_impact(self):
        small = execution_price(50_000.0, "buy", order_size_usd=100)
        large = execution_price(50_000.0, "buy", order_size_usd=500_000)
        assert large > small

    def test_symmetry(self):
        buy = execution_price(50_000.0, "buy", preset="ZERO", order_size_usd=0)
        sell = execution_price(50_000.0, "sell", preset="ZERO", order_size_usd=0)
        assert buy == pytest.approx(50_000.0)
        assert sell == pytest.approx(50_000.0)

class TestTradeCost:
    def test_taker_cost(self):
        cost = trade_cost(10_000.0, "BINANCE_FUTURES_TAKER")
        assert cost == pytest.approx(5.0)

    def test_maker_fallback(self):
        cost = trade_cost(10_000.0, "BINANCE_FUTURES_TAKER", is_maker=True)
        assert cost == pytest.approx(2.0)

    def test_zero_cost(self):
        cost = trade_cost(10_000.0, "ZERO")
        assert cost == pytest.approx(0.0)

class TestAnnualFundingDrag:
    def test_default(self):
        drag = annual_funding_drag()
        assert drag == pytest.approx(0.0001 * 3 * 365)

    def test_custom_rate(self):
        drag = annual_funding_drag(avg_8h_rate=0.0003, long_bias_fraction=0.5)
        assert drag == pytest.approx(0.0003 * 3 * 365 * 0.5)

    def test_zero_rate(self):
        assert annual_funding_drag(avg_8h_rate=0.0) == pytest.approx(0.0)

class TestLiquidationPrice:
    def test_long_liquidation(self):
        entry = 50_000.0
        leverage = 3.0
        units = 0.1
        notional = entry * units
        margin = notional / leverage

        liq = compute_liquidation_price(entry, leverage, "long", margin, units)
        assert liq < entry
        assert liq > 0

    def test_short_liquidation(self):
        entry = 50_000.0
        leverage = 3.0
        units = 0.1
        notional = entry * units
        margin = notional / leverage

        liq = compute_liquidation_price(entry, leverage, "short", margin, units)
        assert liq > entry

    def test_higher_leverage_closer_liquidation(self):
        entry = 50_000.0
        units = 0.1

        liq_3x = compute_liquidation_price(
            entry, 3.0, "long", entry * units / 3.0, units
        )
        liq_10x = compute_liquidation_price(
            entry, 10.0, "long", entry * units / 10.0, units
        )
        assert liq_10x > liq_3x

    def test_funding_erodes_margin_moves_liquidation(self):
        entry = 50_000.0
        leverage = 3.0
        units = 0.1
        notional = entry * units
        initial_margin = notional / leverage

        liq_fresh = compute_liquidation_price(entry, leverage, "long", initial_margin, units)
        eroded_margin = initial_margin * 0.9
        liq_eroded = compute_liquidation_price(entry, leverage, "long", eroded_margin, units)
        assert liq_eroded > liq_fresh

    def test_recalc_matches_compute(self):
        entry = 50_000.0
        units = 0.1
        margin = entry * units / 3.0
        liq1 = compute_liquidation_price(entry, 3.0, "long", margin, units)
        liq2 = recalc_liquidation_price(entry, "long", margin, units, entry)
        assert liq1 == pytest.approx(liq2, rel=0.01)
