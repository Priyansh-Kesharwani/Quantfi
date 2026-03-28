"""Data quality validation framework for crypto OHLCV data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from engine.crypto.calendar import TIMEFRAME_TO_MS


@dataclass
class DataQualityReport:
    """Result of a data quality check."""

    total_expected_bars: int = 0
    actual_bars: int = 0
    missing_bars: int = 0
    max_consecutive_gap: int = 0
    gap_pct: float = 0.0
    stale_bars: int = 0
    is_acceptable: bool = True
    warnings: List[str] = field(default_factory=list)


class DataQualityValidator:
    """Validates OHLCV data integrity before use in backtesting or live trading."""

    MAX_GAP_PCT = 0.02
    MAX_CONSECUTIVE_GAP = 3
    STALE_VOLUME_THRESHOLD = 0.01

    def validate(self, df: pd.DataFrame, timeframe: str) -> DataQualityReport:
        report = DataQualityReport()

        if df.empty:
            report.is_acceptable = False
            report.warnings.append("Empty DataFrame")
            return report

        report.actual_bars = len(df)

        tf_ms = TIMEFRAME_TO_MS.get(timeframe)
        if tf_ms is None:
            report.warnings.append(f"Unknown timeframe '{timeframe}', skipping gap check")
            return report

        if isinstance(df.index, pd.DatetimeIndex):
            diffs = df.index.to_series().diff().dropna()
            expected_td = pd.Timedelta(milliseconds=tf_ms)
            gaps = diffs[diffs > expected_td * 1.5]
            if len(gaps) > 0:
                gap_bars = (gaps / expected_td).astype(int) - 1
                report.missing_bars = int(gap_bars.sum())
                report.max_consecutive_gap = int(gap_bars.max())

            total_span_bars = int(
                (df.index[-1] - df.index[0]) / expected_td
            ) + 1
            report.total_expected_bars = total_span_bars
        else:
            report.total_expected_bars = report.actual_bars

        if report.total_expected_bars > 0:
            report.gap_pct = report.missing_bars / report.total_expected_bars

        if "volume" in df.columns:
            vol = df["volume"]
            rolling_mean = vol.rolling(20, min_periods=1).mean()
            stale_mask = (vol < rolling_mean * self.STALE_VOLUME_THRESHOLD) & (rolling_mean > 0)
            report.stale_bars = int(stale_mask.sum())
            if report.stale_bars > 0:
                report.warnings.append(
                    f"{report.stale_bars} bars with volume < 1% of 20-bar mean"
                )

        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    report.warnings.append(f"{nan_count} NaN values in '{col}'")

        if report.gap_pct > self.MAX_GAP_PCT:
            report.is_acceptable = False
            report.warnings.append(
                f"Gap percentage {report.gap_pct:.1%} exceeds max {self.MAX_GAP_PCT:.0%}"
            )
        if report.max_consecutive_gap > self.MAX_CONSECUTIVE_GAP:
            report.is_acceptable = False
            report.warnings.append(
                f"Max consecutive gap {report.max_consecutive_gap} exceeds max {self.MAX_CONSECUTIVE_GAP}"
            )

        return report

    @staticmethod
    def mark_untradeable(df: pd.DataFrame) -> pd.Series:
        """Return a boolean series: True = bar is tradeable, False = bar has bad data.

        Bars with NaN OHLC values are marked untradeable (NOT forward-filled).
        """
        ohlc_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
        if not ohlc_cols:
            return pd.Series(True, index=df.index)
        return ~df[ohlc_cols].isna().any(axis=1)


def align_funding_to_bars(
    funding_df: pd.DataFrame,
    bar_index: pd.DatetimeIndex,
    timeframe: str,
) -> pd.Series:
    """Align 8h funding rates to arbitrary bar timestamps.

    Bars that contain a funding settlement get the actual rate;
    bars between settlements get 0.0.
    """
    if funding_df.empty:
        return pd.Series(0.0, index=bar_index, name="fundingRate")

    tf_ms = TIMEFRAME_TO_MS.get(timeframe, 3_600_000)
    tolerance = pd.Timedelta(milliseconds=tf_ms)

    funding = funding_df.copy()
    if "timestamp" in funding.columns and not isinstance(funding.index, pd.DatetimeIndex):
        funding = funding.set_index("timestamp")
    funding = funding.sort_index()

    bar_df = pd.DataFrame(index=bar_index)
    merged = pd.merge_asof(
        bar_df,
        funding[["fundingRate"]],
        left_index=True,
        right_index=True,
        direction="backward",
        tolerance=tolerance,
    )
    return merged["fundingRate"].fillna(0.0)
