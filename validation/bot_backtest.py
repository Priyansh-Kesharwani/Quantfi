"""
Bot backtest: entry on score threshold, exit via TBL (kappa_tp, kappa_sl, T_max).

Produces the same result shape as PortfolioSimulator.run() for GT-Score/DSR.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class _Position:
    symbol: str
    units: float
    entry_price: float
    entry_date: date
    entry_bar: int
    atr_at_entry: float
    cost_class: str

    def bars_held(self, current_bar: int) -> int:
        return current_bar - self.entry_bar


def _execution_price(raw_price: float, side: str, slippage_bps: float = 5.0) -> float:
    if slippage_bps <= 0:
        return raw_price
    mult = 1 + (slippage_bps / 10_000) * (1 if side == "buy" else -1)
    return raw_price * mult


def _trade_cost(notional: float, cost_class: str, cost_free: bool = False) -> float:
    if cost_free:
        return 0.0
    from backtester.portfolio_simulator import COST_PRESETS
    cfg = COST_PRESETS.get(cost_class, COST_PRESETS["US_EQ_FROM_IN"])
    pct_cost = notional * (cfg["bps_round_trip"] / 20_000)
    fixed = cfg["fixed_per_trade_inr"] / 2.0
    return pct_cost + fixed


def run_bot_backtest(
    date_index: pd.DatetimeIndex,
    assets: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run backtest with TBL exit (tp/sl/time in bars). Same result shape as PortfolioSimulator.run().

    Config: entry_score_threshold, max_positions, kappa_tp, kappa_sl, T_max (bars),
    initial_capital, slippage_bps, cost_free.
    """
    t0 = time.time()
    entry_threshold = float(config.get("entry_score_threshold", 70.0))
    max_positions = int(config.get("max_positions", 10))
    kappa_tp = float(config.get("kappa_tp", 1.5))
    kappa_sl = float(config.get("kappa_sl", 1.0))
    T_max_bars = int(config.get("T_max", 20))
    initial_capital = float(config.get("initial_capital", 100_000.0))
    slippage_bps = 0.0 if config.get("cost_free") else float(config.get("slippage_bps", 5.0))
    cost_free = bool(config.get("cost_free", False))
    min_notional = float(config.get("min_position_notional", 3_000.0))

    symbols = list(assets.keys())
    if not symbols:
        return {"equity_curve": [], "benchmarks": {}, "trades": [], "warnings": ["No assets"]}

    n = len(date_index)
    cash = initial_capital
    positions: Dict[str, _Position] = {}
    trade_log: List[Dict[str, Any]] = []
    snapshots: List[Dict[str, Any]] = []
    warnings: List[str] = []

    sim_start = 0
    sim_end = n - 1
    for sym in symbols:
        ad = assets[sym]
        fvi = getattr(ad, "first_valid_idx", 0)
        if fvi > sim_start:
            sim_start = max(sim_start, fvi)

    def _current_equity(c: float, pos_dict: Dict[str, _Position], bar_idx: int) -> float:
        eq = c
        for s, p in pos_dict.items():
            ad = assets[s]
            idx = min(bar_idx, len(ad.close) - 1)
            idx = max(idx, 0)
            eq += p.units * ad.close[idx]
        return eq

    for t in range(sim_start, sim_end + 1):
        dt = date_index[t].date() if hasattr(date_index[t], "date") else date_index[t]

        # ── 1. Execute pending exits at today's open ─────────
        to_exit: List[str] = []
        for sym, pos in list(positions.items()):
            ad = assets[sym]
            high_t = float(ad.high[t])
            low_t = float(ad.low[t])
            close_t = float(ad.close[t])
            atr = pos.atr_at_entry if pos.atr_at_entry > 0 else close_t * 0.02
            tp_dist = kappa_tp * atr
            sl_dist = kappa_sl * atr
            upper = pos.entry_price + tp_dist
            lower = pos.entry_price - sl_dist
            bars_held = pos.bars_held(t)
            if high_t >= upper:
                to_exit.append((sym, "tp"))
            elif low_t <= lower:
                to_exit.append((sym, "sl"))
            elif bars_held >= T_max_bars:
                to_exit.append((sym, "time"))

        for sym, reason in to_exit:
            pos = positions.pop(sym)
            ad = assets[sym]
            raw_p = float(ad.open[t])
            exec_p = _execution_price(raw_p, "sell", slippage_bps)
            notional = pos.units * exec_p
            cost = _trade_cost(notional, pos.cost_class, cost_free)
            proceeds = notional - cost
            pnl = proceeds - (pos.units * pos.entry_price)
            hold_days = (dt - pos.entry_date).days if hasattr(dt, "__sub__") else pos.bars_held(t)
            cash += proceeds
            trade_log.append({
                "date": str(dt), "symbol": sym, "side": "EXIT",
                "units": pos.units, "price": round(exec_p, 4),
                "notional": round(notional, 2), "cost": round(cost, 2),
                "score": round(ad.score[t] if t < len(ad.score) else 0, 1),
                "exit_reason": reason, "pnl": round(pnl, 2), "holding_days": hold_days,
                "slippage": round(exec_p - raw_p, 6),
                "post_trade_equity": round(_current_equity(cash, positions, t), 2),
            })

        # ── 2. Entry: score > threshold at close of t-1, execute at t open ─────────
        entry_candidates: List[Tuple[str, float]] = []
        for sym in symbols:
            if sym in positions:
                continue
            ad = assets[sym]
            if t < getattr(ad, "first_valid_idx", 0):
                continue
            if not getattr(ad, "tradeable", np.ones(n))[t]:
                continue
            score = float(ad.score[t]) if t < len(ad.score) else np.nan
            if not np.isnan(score) and score >= entry_threshold:
                entry_candidates.append((sym, score))

        entry_candidates.sort(key=lambda x: x[1], reverse=True)
        slots = max_positions - len(positions)
        entry_candidates = entry_candidates[:max(0, slots)]

        if entry_candidates and cash > min_notional * 1.1:
            allocatable = cash * 0.95
            total_s = sum(s for _, s in entry_candidates)
            for sym, score in entry_candidates:
                w = score / total_s if total_s > 0 else 1.0 / len(entry_candidates)
                alloc = min(allocatable * w, cash - min_notional * 0.1)
                if alloc < min_notional:
                    continue
                ad = assets[sym]
                raw_p = float(ad.open[t])
                exec_p = _execution_price(raw_p, "buy", slippage_bps)
                cost = _trade_cost(alloc, ad.cost_class, cost_free)
                net = alloc - cost
                if net <= 0 or exec_p <= 0:
                    continue
                units = net / exec_p
                if alloc > cash:
                    alloc = cash
                    units = (alloc - cost) / exec_p
                cash -= (units * exec_p + cost)
                atr_entry = float(ad.atr[t]) if t < len(ad.atr) and not np.isnan(ad.atr[t]) else raw_p * 0.02
                positions[sym] = _Position(
                    symbol=sym, units=units, entry_price=exec_p,
                    entry_date=dt, entry_bar=t, atr_at_entry=atr_entry,
                    cost_class=ad.cost_class,
                )
                trade_log.append({
                    "date": str(dt), "symbol": sym, "side": "ENTRY",
                    "units": round(units, 6), "price": round(exec_p, 4),
                    "notional": round(units * exec_p, 2), "cost": round(cost, 2),
                    "score": round(score, 1), "exit_reason": "", "pnl": 0.0, "holding_days": 0,
                    "slippage": round(exec_p - raw_p, 6), "post_trade_equity": 0.0,
                })

        # ── 3. Mark-to-market ─────────────────────────────────
        equity = _current_equity(cash, positions, t)
        invested_pct = ((equity - cash) / equity * 100) if equity > 0 else 0
        snapshots.append({
            "date": str(dt), "equity": round(equity, 2), "cash": round(cash, 2),
            "n_positions": len(positions), "invested_pct": round(invested_pct, 1),
        })

    # Force-close at end
    for sym, pos in list(positions.items()):
        ad = assets[sym]
        raw_p = float(ad.close[sim_end])
        exec_p = _execution_price(raw_p, "sell", slippage_bps)
        notional = pos.units * exec_p
        cost = _trade_cost(notional, pos.cost_class, cost_free)
        proceeds = notional - cost
        pnl = proceeds - (pos.units * pos.entry_price)
        cash += proceeds
        final_dt = date_index[sim_end].date() if hasattr(date_index[sim_end], "date") else date_index[sim_end]
        hold_days = (final_dt - pos.entry_date).days if hasattr(final_dt, "__sub__") else 0
        trade_log.append({
            "date": str(final_dt), "symbol": sym, "side": "EXIT",
            "units": pos.units, "price": round(exec_p, 4),
            "notional": round(notional, 2), "cost": round(cost, 2),
            "score": round(ad.score[sim_end] if sim_end < len(ad.score) else 0, 1),
            "exit_reason": "end_of_sim", "pnl": round(pnl, 2), "holding_days": hold_days,
            "slippage": round(exec_p - raw_p, 6), "post_trade_equity": round(cash, 2),
        })
    positions.clear()

    elapsed = round(time.time() - t0, 2)
    if not snapshots:
        return {"equity_curve": [], "benchmarks": {}, "trades": trade_log, "warnings": warnings}

    equities = np.array([s["equity"] for s in snapshots])
    dates_str = [s["date"] for s in snapshots]
    daily_returns = np.diff(equities) / (equities[:-1] + 1e-12)
    daily_returns = daily_returns[~np.isnan(daily_returns)]

    total_return_pct = (equities[-1] / equities[0] - 1) * 100
    n_days = len(equities)
    n_years = n_days / 252
    cagr = ((equities[-1] / equities[0]) ** (1 / max(n_years, 0.01)) - 1) * 100
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0.0
    neg_ret = daily_returns[daily_returns < 0]
    sortino = (daily_returns.mean() / neg_ret.std() * np.sqrt(252)) if len(neg_ret) > 0 and neg_ret.std() > 0 else 0.0
    running_max = np.maximum.accumulate(equities)
    max_dd = float(np.min((equities - running_max) / (running_max + 1e-12) * 100))
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    exits = [x for x in trade_log if x["side"] == "EXIT"]
    total_trades = len(exits)
    win_rate = len([x for x in exits if x["pnl"] > 0]) / total_trades * 100 if total_trades > 0 else 0
    avg_holding = np.mean([x["holding_days"] for x in exits]) if exits else 0
    total_costs = sum(x["cost"] for x in trade_log)
    cost_drag = total_costs / initial_capital * 100
    time_in_market = np.mean([s["invested_pct"] for s in snapshots]) if snapshots else 0
    exit_reasons: Dict[str, int] = {}
    for x in exits:
        r = x.get("exit_reason") or "unknown"
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    ec = [{"date": s["date"], "equity": s["equity"], "cash": s["cash"],
          "n_positions": s["n_positions"], "invested_pct": s["invested_pct"]} for s in snapshots]
    if len(ec) > 500:
        step = max(1, len(ec) // 500)
        ec = ec[::step]
        if ec[-1]["date"] != snapshots[-1]["date"]:
            ec.append({**snapshots[-1]})

    # Benchmarks (buy-and-hold)
    bnh_alloc = initial_capital / len(symbols)
    bnh_units = {}
    for sym in symbols:
        ad = assets[sym]
        p = ad.close[sim_start]
        if p > 0 and not np.isnan(p):
            bnh_units[sym] = bnh_alloc / p
        else:
            bnh_units[sym] = 0
    bnh_equity = []
    for t in range(sim_start, sim_end + 1):
        val = sum(bnh_units[s] * assets[s].close[t] for s in symbols)
        bnh_equity.append(val)
    bnh_equity = np.array(bnh_equity)
    step_b = max(1, len(bnh_equity) // 500)
    bnh_curve = [{"date": dates_str[i], "equity": round(float(bnh_equity[i]), 2)}
                 for i in range(0, len(bnh_equity), step_b)]
    if bnh_curve and bnh_curve[-1]["date"] != dates_str[-1]:
        bnh_curve.append({"date": dates_str[-1], "equity": round(float(bnh_equity[-1]), 2)})

    return {
        "config": config,
        "total_return_pct": round(total_return_pct, 2),
        "cagr_pct": round(cagr, 2),
        "sharpe_ratio": round(float(sharpe), 3),
        "sortino_ratio": round(float(sortino), 3),
        "calmar_ratio": round(float(calmar), 3),
        "max_drawdown_pct": round(max_dd, 2),
        "total_trades": total_trades,
        "win_rate": round(win_rate, 1),
        "avg_holding_days": round(float(avg_holding), 1),
        "time_in_market_pct": round(float(time_in_market), 1),
        "total_costs": round(total_costs, 2),
        "cost_drag_pct": round(cost_drag, 2),
        "exit_reasons": exit_reasons,
        "asset_breakdown": {},
        "equity_curve": ec,
        "trades": trade_log,
        "benchmarks": {"buy_and_hold": {"equity_curve": bnh_curve}},
        "data_range": f"{dates_str[0]} → {dates_str[-1]}" if dates_str else "",
        "computation_time_s": elapsed,
        "warnings": warnings,
    }
