"""Tests for crypto.india.tax."""

import pytest

from crypto.india.tax import (
    CESS_RATE,
    FLAT_TAX_RATE,
    TDS_RATE,
    add_tax_to_backtest_metrics,
    compute_after_tax_pnl,
    compute_trade_level_tax,
)


class TestComputeAfterTaxPnl:
    def test_profitable_trade(self):
        result = compute_after_tax_pnl(10_000.0, total_sell_proceeds=50_000.0)
        assert result.tax_30pct == pytest.approx(3_000.0)
        assert result.cess_4pct == pytest.approx(120.0)
        assert result.total_tax == pytest.approx(3_120.0)
        assert result.after_tax_pnl == pytest.approx(6_880.0)
        assert result.tds_collected == pytest.approx(500.0)

    def test_loss_no_tax(self):
        result = compute_after_tax_pnl(-5_000.0)
        assert result.taxable_pnl == 0.0
        assert result.total_tax == 0.0
        assert result.after_tax_pnl == pytest.approx(-5_000.0)

    def test_zero_pnl(self):
        result = compute_after_tax_pnl(0.0)
        assert result.total_tax == 0.0
        assert result.effective_tax_rate == 0.0

    def test_effective_rate(self):
        result = compute_after_tax_pnl(10_000.0)
        expected = FLAT_TAX_RATE + FLAT_TAX_RATE * CESS_RATE
        assert result.effective_tax_rate == pytest.approx(expected)


class TestTradeLevelTax:
    def test_no_loss_offset(self):
        trades_pnl = [1_000.0, -500.0, 2_000.0]
        trades_proceeds = [10_000.0, 5_000.0, 20_000.0]
        result = compute_trade_level_tax(trades_pnl, trades_proceeds)
        assert result["total_taxable_gains"] == pytest.approx(3_000.0)
        assert result["total_losses_non_deductible"] == pytest.approx(-500.0)
        assert result["tax_30pct"] == pytest.approx(900.0)

    def test_all_losses(self):
        result = compute_trade_level_tax([-100, -200], [1_000, 2_000])
        assert result["total_tax"] == 0.0
        assert result["gross_pnl"] == pytest.approx(-300.0)

    def test_tds_on_proceeds(self):
        result = compute_trade_level_tax([100], [10_000])
        assert result["tds_collected"] == pytest.approx(100.0)


class TestAddTaxToMetrics:
    def test_augments_metrics(self):
        metrics = {"initial_capital": 10_000, "total_return_pct": 25.0}
        updated = add_tax_to_backtest_metrics(
            metrics,
            trades_pnl=[1_000, 500, -200],
            trades_proceeds=[5_000, 3_000, 2_000],
        )
        assert "india_tax" in updated
        assert "after_tax_return_pct" in updated
        assert updated["after_tax_return_pct"] > 0
