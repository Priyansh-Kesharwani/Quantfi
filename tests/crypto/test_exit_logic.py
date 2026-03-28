"""Tests for exit logic: patience, no stress-exit, trailing stop widening."""

from __future__ import annotations

import warnings
from datetime import datetime

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from engine.crypto.models import FuturesPosition
from engine.crypto.services.backtest_service import CryptoBacktestConfig, CryptoBacktestService


def _make_pos(direction="long", entry_price=100.0, leverage=3.0):
    return FuturesPosition(
        symbol="BTC/USDT:USDT",
        direction=direction,
        units=1.0,
        entry_price=entry_price,
        leverage=leverage,
        margin=entry_price / leverage,
        liquidation_price=entry_price * (1 - 1 / leverage) if direction == "long" else entry_price * (1 + 1 / leverage),
        entry_time=datetime(2024, 1, 1),
        peak_price=entry_price,
    )


class TestNoForcedStressExit:

    def test_no_forced_stress_exit(self):
        """Position survives STRESS regime if score and trailing stop are fine."""
        pos = _make_pos(direction="long", entry_price=100.0)
        pos.peak_price = 105.0
        cfg = CryptoBacktestConfig(exit_threshold=15.0, atr_trail_mult=4.0, max_holding_bars=336)
        reason, _ = CryptoBacktestService._check_exit(
            pos, score=40.0, regime="STRESS", atr_val=2.0,
            close=103.0, bars_held=10, config=cfg, score_below_count=0,
        )
        assert reason is None, f"Should not exit in STRESS with good score: {reason}"

    def test_stress_tightens_trailing_stop(self):
        """During STRESS, trailing stop is 0.5x normal ATR mult (tighter)."""
        pos = _make_pos(direction="long", entry_price=100.0)
        pos.peak_price = 110.0
        cfg = CryptoBacktestConfig(atr_trail_mult=4.0, exit_threshold=5.0)

        reason_normal, _ = CryptoBacktestService._check_exit(
            pos, score=20.0, regime="TRENDING", atr_val=2.0,
            close=102.5, bars_held=10, config=cfg, score_below_count=0,
        )
        reason_stress, _ = CryptoBacktestService._check_exit(
            pos, score=20.0, regime="STRESS", atr_val=2.0,
            close=102.5, bars_held=10, config=cfg, score_below_count=0,
        )
        assert reason_normal is None, "Normal should NOT trigger trailing stop at 102.5"
        assert reason_stress is not None, "Stress should trigger (tighter stop) at 102.5"


class TestScoreExitPatience:

    def test_score_exit_patience(self):
        """Score must stay below threshold for `patience` consecutive bars."""
        pos = _make_pos()
        cfg = CryptoBacktestConfig(exit_threshold=15.0, score_exit_patience=3, atr_trail_mult=4.0)

        reason1, count1 = CryptoBacktestService._check_exit(
            pos, score=5.0, regime="TRENDING", atr_val=2.0,
            close=100.0, bars_held=10, config=cfg, score_below_count=0,
        )
        assert reason1 is None
        assert count1 == 1

        reason2, count2 = CryptoBacktestService._check_exit(
            pos, score=5.0, regime="TRENDING", atr_val=2.0,
            close=100.0, bars_held=11, config=cfg, score_below_count=count1,
        )
        assert reason2 is None
        assert count2 == 2

        reason3, count3 = CryptoBacktestService._check_exit(
            pos, score=5.0, regime="TRENDING", atr_val=2.0,
            close=100.0, bars_held=12, config=cfg, score_below_count=count2,
        )
        assert reason3 is not None
        assert "score_exit" in reason3


class TestTrailingStop:

    def test_trailing_stop_at_4_atr(self):
        pos = _make_pos(direction="long", entry_price=100.0)
        pos.peak_price = 110.0
        cfg = CryptoBacktestConfig(atr_trail_mult=4.0, exit_threshold=5.0)

        reason_in, _ = CryptoBacktestService._check_exit(
            pos, score=20.0, regime="TRENDING", atr_val=2.0,
            close=102.5, bars_held=10, config=cfg, score_below_count=0,
        )
        assert reason_in is None

        reason_out, _ = CryptoBacktestService._check_exit(
            pos, score=20.0, regime="TRENDING", atr_val=2.0,
            close=101.5, bars_held=10, config=cfg, score_below_count=0,
        )
        assert reason_out == "trailing_stop_long"


class TestMaxHolding:

    def test_max_holding_336_bars(self):
        pos = _make_pos()
        cfg = CryptoBacktestConfig(max_holding_bars=336)
        reason_before, _ = CryptoBacktestService._check_exit(
            pos, score=50.0, regime="TRENDING", atr_val=2.0,
            close=100.0, bars_held=335, config=cfg, score_below_count=0,
        )
        assert reason_before is None

        reason_at, _ = CryptoBacktestService._check_exit(
            pos, score=50.0, regime="TRENDING", atr_val=2.0,
            close=100.0, bars_held=336, config=cfg, score_below_count=0,
        )
        assert reason_at is not None
        assert "max_bars" in reason_at
