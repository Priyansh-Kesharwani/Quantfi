"""IStrategy protocol and shared types for all crypto strategies."""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

import pandas as pd

from engine.crypto.models import CryptoTradeRecord, OrderIntent


@runtime_checkable
class IStrategy(Protocol):
    """All strategies implement this protocol."""

    def on_bar(
        self,
        bar: pd.Series,
        bar_idx: int,
        regime: str,
        score: float,
        account_equity: float,
    ) -> List[OrderIntent]:
        ...

    def on_fill(self, trade: CryptoTradeRecord) -> None:
        ...

    @property
    def name(self) -> str:
        ...


def compute_first_valid_bar(
    regime_warmup: int = 504,
    z_max_window: int = 100,
    compression_window: int = 252,
    ic_window: int = 252,
    ic_horizon: int = 20,
    atr_period: int = 14,
    funding_window: int = 504,
    max_data: int = 0,
) -> int:
    """Compute the first bar where all subsystems are warmed up.

    When max_data is provided, caps the warmup at 25% of total data
    so there's enough runway for actual trading.
    """
    raw = max(
        regime_warmup,
        z_max_window + 1,
        compression_window + 1,
        ic_window + ic_horizon,
        atr_period + 1,
        funding_window + 1,
    )
    if max_data > 0:
        return min(raw, max_data // 4)
    return raw
