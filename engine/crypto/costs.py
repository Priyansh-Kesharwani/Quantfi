"""Crypto-specific cost model: fee presets, execution price, market impact, funding drag."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

CRYPTO_COST_PRESETS: Dict[str, Dict[str, float]] = {
    "BINANCE_FUTURES_MAKER": {"fee_pct": 0.0002, "slippage_pct": 0.0001},
    "BINANCE_FUTURES_TAKER": {"fee_pct": 0.0005, "slippage_pct": 0.0002},
    "BINANCE_FUTURES_BNB": {"fee_pct": 0.00018, "slippage_pct": 0.0001},
    "BYBIT_FUTURES_MAKER": {"fee_pct": 0.0001, "slippage_pct": 0.0001},
    "BYBIT_FUTURES_TAKER": {"fee_pct": 0.0006, "slippage_pct": 0.0002},
    "OKX_FUTURES_MAKER": {"fee_pct": 0.0002, "slippage_pct": 0.0001},
    "OKX_FUTURES_TAKER": {"fee_pct": 0.0005, "slippage_pct": 0.0002},
    "PESSIMISTIC": {"fee_pct": 0.001, "slippage_pct": 0.0005},
    "ZERO": {"fee_pct": 0.0, "slippage_pct": 0.0},
}

BINANCE_USDT_M_BRACKETS = [
    (50_000, 0.004, 0),
    (250_000, 0.005, 50),
    (1_000_000, 0.01, 1_300),
    (10_000_000, 0.025, 16_300),
    (20_000_000, 0.05, 266_300),
    (50_000_000, 0.10, 1_266_300),
    (100_000_000, 0.125, 2_516_300),
    (200_000_000, 0.15, 5_016_300),
    (float("inf"), 0.25, 25_016_300),
]


def get_maintenance_margin(position_notional: float) -> tuple[float, float]:
    """Look up MMR and maintenance amount from Binance bracket table.

    Returns (mmr, maintenance_amount).
    """
    for notional_cap, mmr, maint_amount in BINANCE_USDT_M_BRACKETS:
        if position_notional <= notional_cap:
            return mmr, maint_amount
    last = BINANCE_USDT_M_BRACKETS[-1]
    return last[1], last[2]


def execution_price(
    raw_price: float,
    side: str,
    preset: str = "BINANCE_FUTURES_TAKER",
    order_size_usd: float = 0.0,
    reference_volume_usd: float = 1e6,
    slippage_multiplier: float = 1.0,
) -> float:
    """Compute execution price including slippage.

    For retail-scale orders (< 0.1% of reference volume), market impact
    is negligible and only the preset slippage is applied.
    ``slippage_multiplier`` scales the base slippage for stress testing.
    """
    p = CRYPTO_COST_PRESETS.get(preset, CRYPTO_COST_PRESETS["BINANCE_FUTURES_TAKER"])
    total_slip = p["slippage_pct"] * slippage_multiplier

    if order_size_usd > 0 and reference_volume_usd > 0:
        ratio = order_size_usd / reference_volume_usd
        if ratio > 0.001:
            impact = 0.1 * math.pow(ratio, 0.5)
            total_slip += impact

    if side in ("buy", "long_entry", "short_exit"):
        return raw_price * (1.0 + total_slip)
    return raw_price * (1.0 - total_slip)


def trade_cost(
    notional: float,
    preset: str = "BINANCE_FUTURES_TAKER",
    is_maker: bool = False,
) -> float:
    """One-side fee for a trade."""
    p = CRYPTO_COST_PRESETS.get(preset, CRYPTO_COST_PRESETS["BINANCE_FUTURES_TAKER"])
    fee_key = "fee_pct"
    if is_maker and preset.endswith("_TAKER"):
        maker_preset = preset.replace("_TAKER", "_MAKER")
        if maker_preset in CRYPTO_COST_PRESETS:
            p = CRYPTO_COST_PRESETS[maker_preset]
    return notional * p[fee_key]


def annual_funding_drag(
    avg_8h_rate: float = 0.0001,
    long_bias_fraction: float = 1.0,
) -> float:
    """Estimate annualized funding drag.

    Default rate 0.01% per 8h = 3 settlements/day * 365 days * rate.
    """
    return avg_8h_rate * 3.0 * 365.0 * long_bias_fraction


def compute_liquidation_price(
    entry_price: float,
    leverage: float,
    direction: str,
    margin: float,
    units: float,
    position_notional: Optional[float] = None,
) -> float:
    """Compute isolated-margin liquidation price using Binance bracket table.

    Uses actual margin balance (which may have been modified by funding).
    """
    if position_notional is None:
        position_notional = entry_price * units
    mmr, maint_amount = get_maintenance_margin(position_notional)
    maint_margin = position_notional * mmr - maint_amount

    if direction == "long":
        liq = entry_price - (margin - maint_margin) / units
    else:
        liq = entry_price + (margin - maint_margin) / units

    return max(0.0, liq)


def recalc_liquidation_price(
    position_entry_price: float,
    position_direction: str,
    position_margin: float,
    position_units: float,
    mark_price: float,
) -> float:
    """Dynamically recompute liquidation price using current margin and mark price."""
    current_notional = mark_price * position_units
    mmr, maint_amount = get_maintenance_margin(current_notional)
    maint_margin = current_notional * mmr - maint_amount

    if position_direction == "long":
        liq = position_entry_price - (position_margin - maint_margin) / position_units
    else:
        liq = position_entry_price + (position_margin - maint_margin) / position_units

    return max(0.0, liq)
