"""Allocation engine facade -- re-exports from portfolio_simulator for organized imports.

This module exists to provide a clean import path for the allocation
subsystem without duplicating code that is tightly coupled with the
core portfolio simulator.
"""

from backtester.portfolio_simulator import (
    AllocationConfig,
    AllocationEngine,
    RegimeHysteresis,
    RISK_ON,
    RISK_OFF,
    compute_target_weights,
)

__all__ = [
    "AllocationConfig",
    "AllocationEngine",
    "RegimeHysteresis",
    "RISK_ON",
    "RISK_OFF",
    "compute_target_weights",
]
