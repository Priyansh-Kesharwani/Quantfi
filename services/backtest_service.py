from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd

from domain.protocols import IPriceProvider, IBacktestEngine


class BacktestService:
    def __init__(
        self,
        price_provider: IPriceProvider,
        backtest_engine: IBacktestEngine,
        config: Any,
    ) -> None:
        self._price_provider = price_provider
        self._backtest_engine = backtest_engine
        self._config = config
        self._padding_days = getattr(config, "backtest_padding_days", 300)

    def run_backtest(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        dca_amount: float,
        dca_cadence: str,
        buy_dip_threshold: Optional[float] = None,
    ) -> Any:
        if buy_dip_threshold is None:
            buy_dip_threshold = getattr(
                self._config,
                "backtest_buy_dip_threshold_default",
                60.0,
            )
        from backend.models import BacktestConfig

        config = BacktestConfig(
            symbol=symbol.upper(),
            start_date=start_date,
            end_date=end_date,
            dca_amount=dca_amount,
            dca_cadence=dca_cadence,
            buy_dip_threshold=buy_dip_threshold,
        )
        fetch_start = start_date - timedelta(days=self._padding_days)
        df = self._price_provider.fetch_historical_data(
            config.symbol,
            period="max",
            start_date=fetch_start,
            end_date=end_date + timedelta(days=1),
        )
        if df is None or df.empty:
            raise ValueError(
                f"No historical data available for {config.symbol}"
            )
        return self._backtest_engine.run_backtest(config, df)
