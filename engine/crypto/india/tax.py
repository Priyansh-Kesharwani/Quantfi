"""India crypto tax computation: Section 115BBH (30% flat) + 4% cess + Section 194S (1% TDS)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

FLAT_TAX_RATE = 0.30
CESS_RATE = 0.04
TDS_RATE = 0.01


@dataclass
class IndiaTaxResult:
    """Result of India crypto tax computation."""

    gross_pnl: float
    taxable_pnl: float
    tax_30pct: float
    cess_4pct: float
    total_tax: float
    tds_collected: float
    effective_tax_rate: float
    after_tax_pnl: float


def compute_after_tax_pnl(
    gross_pnl: float,
    total_sell_proceeds: float = 0.0,
) -> IndiaTaxResult:
    """Compute after-tax PnL under India's crypto tax regime.

    - Section 115BBH: 30% flat tax on gains (no offset of losses against other income)
    - 4% Health & Education Cess on tax
    - Section 194S: 1% TDS on sell proceeds (credit against final tax)
    - Losses from crypto cannot offset gains from crypto (each transfer taxed independently)
    """
    taxable = max(0.0, gross_pnl)
    tax_30 = taxable * FLAT_TAX_RATE
    cess = tax_30 * CESS_RATE
    total_tax = tax_30 + cess

    tds = total_sell_proceeds * TDS_RATE

    effective_rate = total_tax / gross_pnl if gross_pnl > 0 else 0.0

    return IndiaTaxResult(
        gross_pnl=gross_pnl,
        taxable_pnl=taxable,
        tax_30pct=tax_30,
        cess_4pct=cess,
        total_tax=total_tax,
        tds_collected=tds,
        effective_tax_rate=effective_rate,
        after_tax_pnl=gross_pnl - total_tax,
    )


def compute_trade_level_tax(
    trades_pnl: List[float],
    trades_proceeds: List[float],
) -> Dict[str, float]:
    """Compute aggregate India tax across a list of trades.

    Under Section 115BBH, each profitable trade is taxed at 30% + cess.
    Losses from one trade CANNOT offset gains from another.
    """
    total_taxable = sum(max(0.0, pnl) for pnl in trades_pnl)
    total_loss = sum(min(0.0, pnl) for pnl in trades_pnl)
    gross_pnl = sum(trades_pnl)
    total_proceeds = sum(trades_proceeds)

    tax_30 = total_taxable * FLAT_TAX_RATE
    cess = tax_30 * CESS_RATE
    total_tax = tax_30 + cess
    tds = total_proceeds * TDS_RATE

    return {
        "gross_pnl": gross_pnl,
        "total_taxable_gains": total_taxable,
        "total_losses_non_deductible": total_loss,
        "tax_30pct": tax_30,
        "cess_4pct": cess,
        "total_tax": total_tax,
        "tds_collected": tds,
        "after_tax_pnl": gross_pnl - total_tax,
        "effective_rate_on_gross": total_tax / gross_pnl if gross_pnl > 0 else 0.0,
    }


def add_tax_to_backtest_metrics(
    metrics: Dict,
    trades_pnl: List[float],
    trades_proceeds: List[float],
) -> Dict:
    """Augment backtest metrics with India after-tax figures."""
    tax = compute_trade_level_tax(trades_pnl, trades_proceeds)
    metrics["india_tax"] = tax
    metrics["after_tax_return_pct"] = (
        tax["after_tax_pnl"] / metrics.get("initial_capital", 10_000) * 100
        if metrics.get("initial_capital", 0) > 0
        else 0.0
    )
    return metrics
