from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backend.core.protocols import IPriceProvider, IBacktestEngine
from backend.services.indicator_service import TechnicalIndicators
from backend.models import BacktestConfig, BacktestResult, EquityPoint

logger = logging.getLogger(__name__)


def _default_fx_rate() -> float:
    from backend.core.config import get_backend_config
    return get_backend_config().fx_fallback_usd_inr


class BacktestEngine:

    def __init__(self, fx_provider=None):
        self._fx_provider = fx_provider

    def _get_fx_rate(self) -> float:
        if self._fx_provider is not None:
            rate = self._fx_provider.fetch_usd_inr_rate()
            if rate is not None:
                return rate
        return _default_fx_rate()

    @classmethod
    def _compute_rolling_scores(
        cls, df: pd.DataFrame, fx_rate: Optional[float] = None,
    ) -> pd.Series:
        from backend.services.scoring_service import ScoringEngine

        if fx_rate is None:
            fx_rate = _default_fx_rate()
        n = len(df)
        scores = pd.Series(50.0, index=df.index, dtype=float)

        if n < 200:
            logger.warning(
                "Only %d rows — need ≥200 for full indicator suite; "
                "first 200 rows will use partial indicators",
                n,
            )

        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        sma_50 = TechnicalIndicators.calculate_sma(close, 50)
        sma_200 = TechnicalIndicators.calculate_sma(close, 200)
        rsi_14 = TechnicalIndicators.calculate_rsi(close, 14)

        macd_data = TechnicalIndicators.calculate_macd(close)
        macd_line = macd_data["macd"]
        macd_hist = macd_data["histogram"]

        bb = TechnicalIndicators.calculate_bollinger_bands(close)
        bb_lower = bb["lower"]
        bb_upper = bb["upper"]

        atr_14 = TechnicalIndicators.calculate_atr(high, low, close, 14)
        atr_pctl = TechnicalIndicators.calculate_atr_percentile(
            atr_14, min(252, max(30, n // 4))
        )

        z20 = TechnicalIndicators.calculate_z_score(close, 20)
        z50 = TechnicalIndicators.calculate_z_score(close, 50)
        z100 = TechnicalIndicators.calculate_z_score(close, 100)

        drawdown = TechnicalIndicators.calculate_drawdown(close)
        adx_14 = TechnicalIndicators.calculate_adx(high, low, close, 14)

        for i in range(n):
            ind: Dict = {}

            def _safe(series, idx):
                v = series.iloc[idx]
                return float(v) if pd.notna(v) else None

            ind["sma_200"] = _safe(sma_200, i)
            ind["sma_50"] = _safe(sma_50, i)
            ind["rsi_14"] = _safe(rsi_14, i)
            ind["macd"] = _safe(macd_line, i)
            ind["macd_hist"] = _safe(macd_hist, i)
            ind["bb_lower"] = _safe(bb_lower, i)
            ind["bb_upper"] = _safe(bb_upper, i)
            ind["atr_percentile"] = _safe(atr_pctl, i)
            ind["drawdown_pct"] = _safe(drawdown, i)
            ind["z_score_20"] = _safe(z20, i)
            ind["z_score_50"] = _safe(z50, i)
            ind["z_score_100"] = _safe(z100, i)
            ind["adx_14"] = _safe(adx_14, i)

            current_price = float(close.iloc[i])
            if pd.isna(current_price):
                continue

            score, _, _ = ScoringEngine.calculate_composite_score(
                ind, current_price, fx_rate
            )
            scores.iloc[i] = score

        return scores

    def run_backtest(
        self,
        config: BacktestConfig,
        historical_data: pd.DataFrame,
        scores: Optional[pd.Series] = None,
    ) -> BacktestResult:
        df = historical_data.copy()

        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        data_start = str(df.index.min().date())
        data_end = str(df.index.max().date())
        data_points_total = len(df)

        fx_rate = self._get_fx_rate()

        if scores is not None:
            if hasattr(scores.index, "tz") and scores.index.tz is not None:
                scores = scores.copy()
                scores.index = scores.index.tz_localize(None)
            df["Score"] = scores.reindex(df.index).fillna(50)
        else:
            logger.info("Computing rolling scores for %d rows…", len(df))
            df["Score"] = self._compute_rolling_scores(df, fx_rate)
            logger.info("Rolling scores computed.")

        mask = (df.index >= config.start_date) & (df.index <= config.end_date)
        data = df[mask]

        if data.empty:
            raise ValueError(
                f"No data in range {config.start_date.date()} → {config.end_date.date()}. "
                f"Available: {data_start} → {data_end}"
            )

        dca_dates = self._get_dca_dates(
            config.start_date, config.end_date, config.dca_cadence
        )

        total_invested = 0.0
        total_units = 0.0
        num_regular = 0
        num_dip = 0
        equity_curve: List[EquityPoint] = []

        for dca_date in dca_dates:
            available = data.index[data.index >= dca_date]
            if len(available) == 0:
                continue

            trade_date = available[0]
            price = float(data.loc[trade_date, "Close"])
            score = float(data.loc[trade_date, "Score"])

            if pd.isna(price) or price <= 0:
                continue

            units_bought = config.dca_amount / price
            total_units += units_bought
            total_invested += config.dca_amount
            num_regular += 1

            dip_triggered = False
            if config.buy_dip_threshold and score >= config.buy_dip_threshold:
                dip_amount = config.dca_amount * 0.5
                dip_units = dip_amount / price
                total_units += dip_units
                total_invested += dip_amount
                num_dip += 1
                dip_triggered = True

            equity_curve.append(
                EquityPoint(
                    date=str(trade_date.date()),
                    portfolio_value=round(total_units * price, 2),
                    total_invested=round(total_invested, 2),
                    price=round(price, 2),
                    score=round(score, 1),
                    is_dip_buy=dip_triggered,
                )
            )

        final_price = float(data.iloc[-1]["Close"])
        final_value_usd = total_units * final_price
        final_value_inr = final_value_usd * fx_rate

        total_return_pct = (
            ((final_value_usd - total_invested) / total_invested) * 100
            if total_invested > 0
            else 0
        )

        days = (config.end_date - config.start_date).days
        years = days / 365.25
        annualized_return_pct = (
            (((final_value_usd / total_invested) ** (1 / years)) - 1) * 100
            if years > 0 and total_invested > 0
            else 0
        )

        avg_cost_basis = total_invested / total_units if total_units > 0 else 0

        max_drawdown_pct = 0.0
        if equity_curve:
            pv = pd.Series([p.portfolio_value for p in equity_curve])
            running_max = pv.expanding().max()
            dd = ((pv - running_max) / running_max) * 100
            max_drawdown_pct = float(dd.min()) if len(dd) > 0 else 0.0

        if len(equity_curve) > 500:
            step = max(1, len(equity_curve) // 500)
            equity_curve = equity_curve[::step]
            if equity_curve[-1].date != str(data.index[-1].date()):
                equity_curve.append(
                    EquityPoint(
                        date=str(data.index[-1].date()),
                        portfolio_value=round(final_value_usd, 2),
                        total_invested=round(total_invested, 2),
                        price=round(final_price, 2),
                        score=round(float(data.iloc[-1]["Score"]), 1),
                    )
                )

        return BacktestResult(
            symbol=config.symbol,
            config=config,
            total_invested=round(total_invested, 2),
            total_units=round(total_units, 6),
            final_value_usd=round(final_value_usd, 2),
            final_value_inr=round(final_value_inr, 2),
            total_return_pct=round(total_return_pct, 2),
            annualized_return_pct=round(annualized_return_pct, 2),
            num_regular_dca=num_regular,
            num_dip_buys=num_dip,
            max_drawdown_pct=round(max_drawdown_pct, 2),
            avg_cost_basis=round(avg_cost_basis, 2),
            data_points=data_points_total,
            data_source="yfinance",
            data_start=data_start,
            data_end=data_end,
            equity_curve=equity_curve,
        )

    @staticmethod
    def _get_dca_dates(
        start_date: datetime, end_date: datetime, cadence: str,
    ) -> List[datetime]:
        dates = []
        current = start_date

        if cadence == "weekly":
            while current <= end_date:
                dates.append(current)
                current += timedelta(days=7)
        elif cadence == "monthly":
            while current <= end_date:
                dates.append(current)
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1)
                else:
                    try:
                        current = current.replace(month=current.month + 1)
                    except ValueError:
                        current = current.replace(
                            month=current.month + 1, day=min(current.day, 28)
                        )

        return dates


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
                self._config, "backtest_buy_dip_threshold_default", 60.0
            )

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
