"""CryptoBacktestService: full backtest pipeline with baseline comparisons."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from engine.crypto.calendar import (
    annualized_cagr,
    annualized_sharpe,
    annualized_sortino,
    bootstrap_sharpe_pvalue,
    calmar_ratio,
    max_drawdown,
    profit_factor,
)
from engine.crypto.costs import execution_price, trade_cost
from engine.crypto.engines.futures_engine import FuturesEngine, FuturesEngineConfig
from engine.crypto.engines.grid_engine import GridConfig, GridEngine, verify_grid_profitability
from engine.crypto.models import CryptoTradeRecord, FuturesPosition
from engine.crypto.regime.detector import CryptoRegimeConfig, CryptoRegimeDetector
from engine.crypto.scoring.directional_scorer import (
    CryptoDirectionalScorer,
    ScoringConfig,
    verify_score_reachability,
)
from engine.crypto.adapters.data_quality import DataQualityValidator
from engine.crypto.strategies.base import compute_first_valid_bar

logger = logging.getLogger(__name__)

@dataclass
class CryptoBacktestConfig:
    symbol: str = "BTC/USDT:USDT"
    timeframe: str = "1h"
    initial_capital: float = 10_000.0
    strategy_mode: Literal["directional", "grid", "adaptive"] = "adaptive"
    leverage: float = 3.0
    cost_preset: str = "BINANCE_FUTURES_TAKER"

    entry_threshold: float = 30.0
    exit_threshold: float = 15.0
    max_holding_bars: int = 336
    atr_trail_mult: float = 4.0
    kelly_fraction: float = 0.25
    max_risk_per_trade: float = 0.02
    kelly_lookback: int = 100
    score_exit_patience: int = 3

    grid_levels: int = 20
    grid_spacing: str = "geometric"
    grid_order_size: float = 100.0
    atr_multiplier: float = 3.0

    transition_cooldown: int = 3
    slippage_multiplier: float = 1.0

    scoring_config: Optional[ScoringConfig] = None
    regime_config: Optional[CryptoRegimeConfig] = None

    rsi_oversold: float = 25.0
    rsi_overbought: float = 75.0
    compression_window: int = 120
    z_short_weight: float = 0.40
    z_mid_weight: float = 0.35
    vol_threshold: float = 2.0
    vol_scale: float = 0.5
    vol_filter_pctl: float = 0.90
    vol_filter_floor: float = 0.3
    vol_filter_width: float = 0.10
    funding_window: int = 200
    ic_window: int = 120
    ic_horizon: int = 20
    ic_alpha: float = 5.0
    ic_shrink: float = 0.2

class CryptoBacktestService:
    """Runs full backtests for the crypto trading bot."""

    def run(
        self,
        ohlcv: pd.DataFrame,
        config: CryptoBacktestConfig,
        funding_rates: Optional[pd.Series] = None,
        open_interest: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """Full backtest pipeline on provided OHLCV data."""
        dq_report = DataQualityValidator().validate(ohlcv, config.timeframe)
        if not dq_report.is_acceptable:
            raise ValueError(f"Data quality check failed: {dq_report.warnings}")
        if dq_report.warnings:
            for w in dq_report.warnings:
                logger.warning("Data quality: %s", w)

        scoring_cfg = config.scoring_config or ScoringConfig(
            rsi_oversold=config.rsi_oversold,
            rsi_overbought=config.rsi_overbought,
            compression_window=config.compression_window,
            z_short_weight=config.z_short_weight,
            z_mid_weight=config.z_mid_weight,
            vol_threshold=config.vol_threshold,
            vol_scale=config.vol_scale,
            vol_filter_pctl=config.vol_filter_pctl,
            vol_filter_floor=config.vol_filter_floor,
            vol_filter_width=config.vol_filter_width,
            funding_window=config.funding_window,
            ic_window=config.ic_window,
            ic_horizon=config.ic_horizon,
            ic_alpha=config.ic_alpha,
            ic_shrink=config.ic_shrink,
            entry_threshold=config.entry_threshold,
            exit_threshold=config.exit_threshold,
        )

        scorer = CryptoDirectionalScorer(scoring_cfg)
        scores = scorer.compute_with_uniform_weights(ohlcv, funding_rates, open_interest)

        regime_cfg = config.regime_config or CryptoRegimeConfig()
        regime_det = CryptoRegimeDetector(regime_cfg)
        regimes = regime_det.fit_rolling(ohlcv)

        atr = scorer._compute_atr_series(
            ohlcv.get("high", ohlcv["close"]),
            ohlcv.get("low", ohlcv["close"]),
            ohlcv["close"],
            14,
        )

        first_valid = compute_first_valid_bar(
            regime_warmup=regime_cfg.warmup_bars,
            compression_window=config.compression_window,
            ic_window=config.ic_window,
            ic_horizon=config.ic_horizon,
            funding_window=config.funding_window,
            max_data=len(ohlcv),
        )

        scores, regimes, atr, funding_rates = self._validate_inputs(
            ohlcv, scores, regimes, atr, funding_rates, first_valid,
        )

        result = self._run_bar_by_bar(
            ohlcv, scores, regimes, atr, config, first_valid, funding_rates,
        )

        baselines = self._compute_baselines(ohlcv, config, first_valid)
        result["baselines"] = baselines

        reachability = verify_score_reachability(
            scores.iloc[first_valid:], config.entry_threshold
        )
        result["score_reachability"] = reachability

        return result

    @staticmethod
    def _validate_inputs(
        ohlcv: pd.DataFrame,
        scores: pd.Series,
        regimes: pd.Series,
        atr: pd.Series,
        funding_rates: Optional[pd.Series],
        first_valid: int,
    ):
        """Sanitize inputs: align lengths, fill NaN, log warnings."""
        n = len(ohlcv)

        if len(scores) != n:
            logger.warning("scores length %d != ohlcv length %d; reindexing", len(scores), n)
            scores = scores.reindex(ohlcv.index).fillna(0.0)
        if len(regimes) != n:
            logger.warning("regimes length %d != ohlcv length %d; reindexing", len(regimes), n)
            regimes = regimes.reindex(ohlcv.index).fillna("RANGING")
        if len(atr) != n:
            logger.warning("atr length %d != ohlcv length %d; reindexing", len(atr), n)
            atr = atr.reindex(ohlcv.index).fillna(ohlcv["close"] * 0.02)

        nan_pct_scores = scores.isna().mean()
        if nan_pct_scores > 0.05:
            logger.warning("%.1f%% NaN in scores", nan_pct_scores * 100)
        scores = scores.fillna(0.0)

        regimes = regimes.fillna("RANGING")

        nan_pct_atr = atr.isna().mean()
        if nan_pct_atr > 0.05:
            logger.warning("%.1f%% NaN in ATR", nan_pct_atr * 100)
        atr = atr.fillna(ohlcv["close"] * 0.02)

        if funding_rates is not None:
            if len(funding_rates) != n:
                funding_rates = funding_rates.reindex(ohlcv.index).fillna(0.0)
            funding_rates = funding_rates.fillna(0.0)

        if first_valid >= n - 50:
            logger.warning("first_valid=%d leaves only %d bars for trading", first_valid, n - first_valid)

        return scores, regimes, atr, funding_rates

    @staticmethod
    def _regime_to_mode(regime: str) -> str:
        if regime == "TRENDING":
            return "directional"
        elif regime == "RANGING":
            return "grid"
        return "directional"

    def _run_bar_by_bar(
        self,
        ohlcv: pd.DataFrame,
        scores: pd.Series,
        regimes: pd.Series,
        atr: pd.Series,
        config: CryptoBacktestConfig,
        first_valid: int,
        funding_rates: Optional[pd.Series],
    ) -> Dict[str, Any]:
        n = len(ohlcv)
        equity = np.full(n, config.initial_capital)
        capital = config.initial_capital
        trades: List[CryptoTradeRecord] = []
        trade_pnl_history: List[float] = []

        engine = FuturesEngine(FuturesEngineConfig(
            leverage=config.leverage,
            cost_preset=config.cost_preset,
            slippage_multiplier=config.slippage_multiplier,
        ))

        grid_engine = GridEngine(GridConfig(
            num_levels=config.grid_levels,
            spacing=config.grid_spacing,
            order_size_usd=config.grid_order_size,
        ))

        position: Optional[FuturesPosition] = None
        active_mode = config.strategy_mode
        if active_mode == "adaptive":
            active_mode = "directional"
        cooldown = 0
        bars_in_position = 0
        score_below_count = 0

        for i in range(1, n):
            close = float(ohlcv["close"].iloc[i])
            high_val = float(ohlcv.get("high", ohlcv["close"]).iloc[i])
            low_val = float(ohlcv.get("low", ohlcv["close"]).iloc[i])
            open_val = float(ohlcv.get("open", ohlcv["close"]).iloc[i])
            ts = ohlcv.index[i] if isinstance(ohlcv.index, pd.DatetimeIndex) else datetime(2024, 1, 1)

            prev = i - 1
            signal_score = float(scores.iloc[prev]) if not np.isnan(scores.iloc[prev]) else 0.0
            signal_regime = str(regimes.iloc[prev])
            signal_atr = float(atr.iloc[prev]) if prev < len(atr) and not np.isnan(atr.iloc[prev]) else open_val * 0.02

            if position is not None:
                fr = float(funding_rates.iloc[i]) if funding_rates is not None and i < len(funding_rates) else 0.0
                if fr != 0.0:
                    engine.apply_funding_if_due(position, fr, close, ts)

                liq, loss = engine.check_liquidation(position, high_val, low_val)
                if liq:
                    capital += max(0, position.margin - loss)
                    trades.append(CryptoTradeRecord(
                        timestamp=ts, symbol=config.symbol,
                        side="long_exit" if position.direction == "long" else "short_exit",
                        units=position.units, price=position.liquidation_price,
                        notional=position.units * position.liquidation_price,
                        fee=loss * 0.005, slippage=0,
                        funding_paid=position.accumulated_funding,
                        pnl=-loss, exit_reason="liquidation",
                        leverage=position.leverage, bar_idx=i,
                    ))
                    position = None
                    bars_in_position = 0
                    score_below_count = 0
                else:
                    bars_in_position += 1
                    position.update_peak(close)

            if i < first_valid:
                unrealized = position.unrealized_pnl(close) if position else 0
                margin = position.margin if position else 0
                equity[i] = capital + margin + unrealized
                continue

            exec_price = open_val

            if config.strategy_mode == "adaptive":
                if cooldown > 0:
                    cooldown -= 1
                else:
                    target = self._regime_to_mode(signal_regime)
                    if target != active_mode:
                        capital, position, bars_in_position = self._flatten_all(
                            capital, position, grid_engine, engine, exec_price, ts, i,
                            config.symbol, trades,
                        )
                        score_below_count = 0
                        active_mode = target
                        cooldown = config.transition_cooldown

            if active_mode == "directional" or config.strategy_mode == "directional":
                capital, position, bars_in_position, score_below_count = self._directional_step(
                    config, engine, capital, position, bars_in_position,
                    signal_score, signal_regime, signal_atr, exec_price, ts, i,
                    config.symbol, trades, trade_pnl_history,
                    score_below_count,
                )
            elif active_mode == "grid" or config.strategy_mode == "grid":
                capital = self._grid_step(
                    config, grid_engine, capital, close, open_val, high_val,
                    low_val, signal_atr, ts, i, config.symbol, trades, signal_regime,
                )

            unrealized = position.unrealized_pnl(close) if position else 0
            margin = position.margin if position else 0
            grid_unr = 0.0
            if grid_engine.is_active and grid_engine._inventory_units > 0:
                avg_c = grid_engine._inventory_cost / max(grid_engine._inventory_units, 1e-12)
                grid_unr = (close - avg_c) * grid_engine._inventory_units
            equity[i] = capital + margin + unrealized + grid_unr

        if position is not None:
            close_final = float(ohlcv["close"].iloc[-1])
            ts_final = ohlcv.index[-1] if isinstance(ohlcv.index, pd.DatetimeIndex) else datetime(2024, 1, 1)
            trade = engine.close_position(position, close_final, "end_of_backtest", n - 1, ts_final)
            capital += trade.pnl + position.margin
            trades.append(trade)
            position = None

        if grid_engine.is_active:
            close_final = float(ohlcv["close"].iloc[-1])
            ts_final = ohlcv.index[-1] if isinstance(ohlcv.index, pd.DatetimeIndex) else datetime(2024, 1, 1)
            final_bar = len(ohlcv) - 1
            gt = grid_engine.close_all(close_final, ts_final, "end_of_backtest")
            for t in gt:
                t.bar_idx = final_bar
                capital += t.notional - t.fee if t.side == "grid_sell" else 0
                trades.append(t)

        equity[-1] = capital

        _recon_expected = config.initial_capital + sum(t.pnl for t in trades)
        _recon_diff = abs(_recon_expected - capital)
        if _recon_diff > 1.0:
            logger.warning(
                "Reconciliation mismatch: trade_pnl_sum+initial=%.2f vs capital=%.2f (diff=%.2f)",
                _recon_expected, capital, _recon_diff,
            )

        equity_series = pd.Series(equity, index=ohlcv.index)
        metrics = self._compute_analytics(equity_series, trades, config.timeframe)
        metrics["equity_curve"] = equity_series
        metrics["trades"] = trades
        metrics["regimes"] = regimes
        metrics["scores"] = scores
        return metrics

    def _flatten_all(
        self,
        capital: float,
        position: Optional[FuturesPosition],
        grid_engine: GridEngine,
        engine: FuturesEngine,
        close: float,
        ts: datetime,
        bar_idx: int,
        symbol: str,
        trades: List[CryptoTradeRecord],
    ):
        """Flatten directional position and grid for mode switches."""
        if position is not None:
            trade = engine.close_position(position, close, "mode_switch", bar_idx, ts)
            capital += trade.pnl + position.margin
            trades.append(trade)
            position = None

        if grid_engine.is_active:
            gt = grid_engine.close_all(close, ts, "mode_switch")
            for t in gt:
                t.bar_idx = bar_idx
                if t.side == "grid_sell":
                    capital += t.notional - t.fee
                trades.append(t)

        return capital, None, 0

    def _directional_step(
        self,
        config: CryptoBacktestConfig,
        engine: FuturesEngine,
        capital: float,
        position: Optional[FuturesPosition],
        bars_in_pos: int,
        score: float,
        regime: str,
        atr_val: float,
        close: float,
        ts: datetime,
        bar_idx: int,
        symbol: str,
        trades: List[CryptoTradeRecord],
        pnl_history: List[float],
        score_below_count: int,
    ):
        """Execute one directional step: check entry/exit."""
        if position is None:
            if regime != "STRESS" and abs(score) > config.entry_threshold:
                size_frac = self._position_fraction(config, pnl_history, atr_val, close, score)
                alloc = capital * size_frac
                if alloc > 10:
                    direction = "long" if score > 0 else "short"
                    pos = engine.open_position(
                        symbol, direction, alloc, close, bar_idx, ts,
                    )
                    if pos is not None:
                        capital -= pos.margin + pos.accumulated_fees
                        trades.append(CryptoTradeRecord(
                            timestamp=ts, symbol=symbol,
                            side="long_entry" if direction == "long" else "short_entry",
                            units=pos.units, price=pos.entry_price,
                            notional=pos.notional,
                            fee=pos.accumulated_fees, slippage=abs(pos.entry_price - close) * pos.units,
                            funding_paid=0.0, pnl=0.0,
                            exit_reason="", leverage=pos.leverage, bar_idx=bar_idx,
                        ))
                        position = pos
                        bars_in_pos = 0
                        score_below_count = 0
        else:
            exit_reason, score_below_count = self._check_exit(
                position, score, regime, atr_val, close, bars_in_pos, config,
                score_below_count,
            )
            if exit_reason:
                trade = engine.close_position(position, close, exit_reason, bar_idx, ts)
                capital += trade.pnl + position.margin
                trades.append(trade)
                pnl_history.append(trade.pnl)
                position = None
                bars_in_pos = 0
                score_below_count = 0

        return capital, position, bars_in_pos, score_below_count

    def _grid_step(
        self,
        config: CryptoBacktestConfig,
        grid_engine: GridEngine,
        capital: float,
        close: float,
        open_val: float,
        high_val: float,
        low_val: float,
        atr_val: float,
        ts: datetime,
        bar_idx: int,
        symbol: str,
        trades: List[CryptoTradeRecord],
        regime: str,
    ) -> float:
        """Execute one grid step with capital-proportional order sizing."""
        if not grid_engine.is_active:
            order_size = max(10.0, capital * 0.01)
            grid_engine.config.order_size_usd = order_size
            half_range = atr_val * config.atr_multiplier
            if half_range < close * 0.001:
                half_range = close * 0.02
            grid_engine.config.lower_price = close - half_range
            grid_engine.config.upper_price = close + half_range
            success = grid_engine.initialize(close, symbol, initial_equity=capital)
            if not success:
                grid_engine.config.num_levels = max(5, config.grid_levels // 2)
                half_range = close * 0.03
                grid_engine.config.lower_price = close - half_range
                grid_engine.config.upper_price = close + half_range
                grid_engine.initialize(close, symbol, initial_equity=capital)

        if grid_engine.is_active:
            total_equity = capital
            if grid_engine._inventory_units > 0:
                avg_c = grid_engine._inventory_cost / max(grid_engine._inventory_units, 1e-12)
                total_equity += (close - avg_c) * grid_engine._inventory_units
            if total_equity < config.initial_capital * 0.05:
                gt = grid_engine.close_all(close, ts, "equity_stop")
                for t in gt:
                    t.bar_idx = bar_idx
                    if t.side == "grid_sell":
                        capital += t.notional - t.fee
                    trades.append(t)
                return capital

            max_inventory_cost = config.initial_capital * 0.5
            grid_trades = grid_engine.on_bar(open_val, high_val, low_val, close, ts)
            for t in grid_trades:
                t.bar_idx = bar_idx
                if t.side == "grid_buy":
                    cost = t.notional + t.fee
                    if capital < cost or grid_engine._inventory_cost > max_inventory_cost:
                        continue
                    capital -= cost
                elif t.side == "grid_sell":
                    capital += t.notional - t.fee
                trades.append(t)

        return capital

    @staticmethod
    def _position_fraction(
        config: CryptoBacktestConfig,
        pnl_history: List[float],
        atr_val: float,
        close: float,
        score: float = 0.0,
    ) -> float:
        c = config
        base_frac = c.max_risk_per_trade

        if len(pnl_history) >= 30:
            recent = pnl_history[-c.kelly_lookback:]
            wins = [p for p in recent if p > 0]
            losses = [p for p in recent if p < 0]
            if wins and losses:
                p = len(wins) / len(recent)
                b = np.mean(wins) / abs(np.mean(losses))
                q = 1.0 - p
                f_star = max(0.0, (p * b - q) / max(b, 1e-12))
                base_frac = min(f_star * c.kelly_fraction, c.max_risk_per_trade)

        score_scale = min(abs(score) / 100.0, 1.0)
        frac = base_frac * (0.5 + 0.5 * score_scale)
        max_frac = 3.0 / max(c.leverage, 1.0)
        return min(frac, max_frac)

    @staticmethod
    def _check_exit(
        pos: FuturesPosition,
        score: float,
        regime: str,
        atr_val: float,
        close: float,
        bars_held: int,
        config: CryptoBacktestConfig,
        score_below_count: int,
    ) -> tuple[Optional[str], int]:
        if bars_held >= config.max_holding_bars:
            return f"max_bars={config.max_holding_bars}", 0

        atr_mult = config.atr_trail_mult
        if regime == "STRESS":
            atr_mult *= 0.5
        if atr_val > 0 and atr_mult > 0:
            if pos.direction == "long":
                trail = pos.peak_price - atr_mult * atr_val
                if close < trail:
                    return "trailing_stop_long", 0
            else:
                trail = pos.peak_price + atr_mult * atr_val
                if close > trail:
                    return "trailing_stop_short", 0

        if abs(score) < config.exit_threshold:
            score_below_count += 1
            if score_below_count >= config.score_exit_patience:
                return f"score_exit={score:.1f}", 0
        else:
            score_below_count = 0

        return None, score_below_count

    def _compute_analytics(
        self,
        equity: pd.Series,
        trades: List[CryptoTradeRecord],
        timeframe: str,
    ) -> Dict[str, Any]:
        returns = equity.pct_change().fillna(0)
        pnls = pd.Series([t.pnl for t in trades if t.pnl != 0]) if trades else pd.Series(dtype=float)

        return {
            "sharpe": annualized_sharpe(returns, timeframe),
            "sortino": annualized_sortino(returns, timeframe),
            "cagr": annualized_cagr(equity, timeframe),
            "max_drawdown": max_drawdown(equity),
            "calmar": calmar_ratio(equity, timeframe),
            "win_rate": float((pnls > 0).mean()) if len(pnls) > 0 else 0.0,
            "profit_factor": profit_factor(pnls) if len(pnls) > 0 else 0.0,
            "avg_trade_pnl": float(pnls.mean()) if len(pnls) > 0 else 0.0,
            "total_funding": sum(t.funding_paid for t in trades),
            "total_fees": sum(t.fee for t in trades),
            "n_trades": len(trades),
            "avg_holding_bars": self._avg_holding_duration(trades),
            "final_equity": float(equity.iloc[-1]),
            "total_return_pct": float((equity.iloc[-1] / equity.iloc[0] - 1) * 100),
            "sharpe_significance": bootstrap_sharpe_pvalue(
                returns, timeframe, n_bootstrap=2_000,
            ),
        }

    @staticmethod
    def _avg_holding_duration(trades: List[CryptoTradeRecord]) -> float:
        """Compute average holding duration by pairing entry and exit bar indices."""
        entry_bars: List[int] = []
        durations: List[int] = []
        for t in trades:
            if t.side in ("long_entry", "short_entry"):
                entry_bars.append(t.bar_idx)
            elif t.side in ("long_exit", "short_exit") and entry_bars:
                dur = t.bar_idx - entry_bars.pop(0)
                durations.append(max(0, dur))
        return float(np.mean(durations)) if durations else 0.0

    def _compute_baselines(
        self,
        ohlcv: pd.DataFrame,
        config: CryptoBacktestConfig,
        first_valid: int,
    ) -> Dict[str, Dict[str, float]]:
        """Compute buy-and-hold, always-long, and random-entry baselines."""
        close = ohlcv["close"]
        valid_close = close.iloc[first_valid:]

        bh_returns = valid_close.pct_change().fillna(0)
        bh_equity = (1 + bh_returns).cumprod() * config.initial_capital

        liq_floor = -1.0 / max(config.leverage, 1.0)
        al_returns = (valid_close.pct_change().fillna(0) * config.leverage).clip(lower=liq_floor)
        al_equity = (1 + al_returns).cumprod() * config.initial_capital

        rng = np.random.RandomState(42)
        positions = rng.choice([-1, 0, 1], size=len(valid_close), p=[0.3, 0.4, 0.3])
        rand_returns = (valid_close.pct_change().fillna(0) * positions * config.leverage).clip(lower=liq_floor)
        rand_equity = (1 + rand_returns).cumprod() * config.initial_capital

        return {
            "buy_and_hold": {
                "sharpe": annualized_sharpe(bh_returns, config.timeframe),
                "cagr": annualized_cagr(bh_equity, config.timeframe),
                "max_drawdown": max_drawdown(bh_equity),
                "final_equity": float(bh_equity.iloc[-1]),
                "total_return_pct": float((bh_equity.iloc[-1] / config.initial_capital - 1) * 100),
            },
            "always_long": {
                "sharpe": annualized_sharpe(al_returns, config.timeframe),
                "cagr": annualized_cagr(al_equity, config.timeframe),
                "max_drawdown": max_drawdown(al_equity),
                "final_equity": float(al_equity.iloc[-1]),
                "total_return_pct": float((al_equity.iloc[-1] / config.initial_capital - 1) * 100),
            },
            "random_entry": {
                "sharpe": annualized_sharpe(pd.Series(rand_returns.values), config.timeframe),
                "cagr": annualized_cagr(pd.Series(rand_equity.values), config.timeframe),
                "max_drawdown": max_drawdown(pd.Series(rand_equity.values)),
                "final_equity": float(rand_equity.iloc[-1]),
                "total_return_pct": float((rand_equity.iloc[-1] / config.initial_capital - 1) * 100),
            },
        }
