from __future__ import annotations

from typing import Optional, Any
import logging

import yfinance as yf

logger = logging.getLogger(__name__)


class FXAdapter:
    def __init__(self, config: Any) -> None:
        self._config = config
        self._period = getattr(config, "latest_price_period", "5d")
        self._fallback = getattr(config, "fx_fallback_usd_inr", 83.5)

    def fetch_usd_inr_rate(self) -> Optional[float]:
        try:
            ticker = yf.Ticker("USDINR=X")
            df = ticker.history(period=self._period)
            if df is not None and not df.empty:
                return float(df.iloc[-1]["Close"])
            return self._fallback
        except Exception as e:
            logger.error("FX rate error: %s", e)
            return self._fallback
