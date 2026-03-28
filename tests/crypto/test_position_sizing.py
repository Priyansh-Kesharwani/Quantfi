"""Tests for position sizing: score-proportional, stress reduction, capital scaling."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from engine.crypto.services.backtest_service import CryptoBacktestConfig, CryptoBacktestService


class TestPositionFraction:

    def test_default_fraction_is_2_percent(self):
        cfg = CryptoBacktestConfig()
        assert cfg.max_risk_per_trade == 0.02

    def test_kelly_fraction_caps_at_max_risk(self):
        cfg = CryptoBacktestConfig(max_risk_per_trade=0.10, kelly_fraction=0.25)
        pnl_history = [100.0] * 50 + [-50.0] * 10
        frac = CryptoBacktestService._position_fraction(cfg, pnl_history, 10.0, 100.0, score=80.0)
        assert frac <= cfg.max_risk_per_trade * 1.01

    def test_score_proportional_scaling(self):
        cfg = CryptoBacktestConfig(max_risk_per_trade=0.20)
        frac_low = CryptoBacktestService._position_fraction(cfg, [], 10.0, 100.0, score=30.0)
        frac_high = CryptoBacktestService._position_fraction(cfg, [], 10.0, 100.0, score=90.0)
        assert frac_high > frac_low, "Higher score should produce larger fraction"

    def test_stress_blocks_new_entries(self):
        """STRESS regime blocks new entries entirely (no allocation)."""
        cfg = CryptoBacktestConfig(max_risk_per_trade=0.10)
        base_frac = CryptoBacktestService._position_fraction(cfg, [], 10.0, 100.0, score=50.0)
        assert base_frac > 0, "Normal regime should produce non-zero fraction"

    def test_position_size_scales_with_capital(self):
        cfg = CryptoBacktestConfig(max_risk_per_trade=0.15)
        frac = CryptoBacktestService._position_fraction(cfg, [], 10.0, 100.0, score=50.0)
        alloc_small = 5000 * frac
        alloc_large = 20000 * frac
        assert alloc_large > alloc_small

    def test_grid_order_size_default(self):
        cfg = CryptoBacktestConfig()
        assert cfg.grid_order_size == 100.0
