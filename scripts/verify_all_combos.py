#!/usr/bin/env python
"""Comprehensive verification: run all pair/strategy/leverage/timeframe combos,
export trade data, and validate every metric."""

from __future__ import annotations

import csv
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.crypto.synthetic import generate_synthetic_data as _generate_synthetic_data, generate_synthetic_funding as _generate_synthetic_funding
from crypto.regime.detector import CryptoRegimeConfig, CryptoRegimeDetector
from crypto.services.backtest_service import CryptoBacktestConfig, CryptoBacktestService

SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT"]
STRATEGIES = ["directional", "grid", "adaptive"]
LEVERAGES = [2, 5, 10]
TIMEFRAMES = ["1h", "4h", "1d"]

OUT_DIR = ROOT / "validation" / "trade_exports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _freq_for_tf(tf: str) -> str:
    return {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}[tf]

def _n_bars_for_tf(tf: str) -> int:
    return {"1m": 5000, "5m": 3000, "15m": 2000, "1h": 2000, "4h": 1500, "1d": 1000}[tf]

def _warmup_for_n(n: int) -> int:
    return min(500, n // 4)

def _generate_data(n: int, symbol: str, tf: str) -> tuple[pd.DataFrame, pd.Series]:
    params = {
        "BTC/USDT:USDT": {"base": 42000, "vol": 0.015, "drift": 0.0002},
        "ETH/USDT:USDT": {"base": 2500, "vol": 0.020, "drift": 0.0003},
        "SOL/USDT:USDT": {"base": 100, "vol": 0.030, "drift": 0.0004},
        "BNB/USDT:USDT": {"base": 320, "vol": 0.018, "drift": 0.0002},
    }
    p = params.get(symbol, {"base": 1000, "vol": 0.015, "drift": 0.0002})
    sym_hash = sum(ord(c) for c in symbol) % 1000
    rng = np.random.RandomState(42 + sym_hash)

    returns = np.zeros(n)
    i = 0
    while i < n:
        regime_len = max(20, int(rng.geometric(1 / 150)))
        end = min(i + regime_len, n)
        seg = end - i
        regime = rng.choice(["bull", "bear", "range", "volatile"], p=[0.30, 0.20, 0.35, 0.15])
        if regime == "bull":
            returns[i:end] = rng.normal(p["drift"] * 4, p["vol"] * 0.8, seg)
        elif regime == "bear":
            returns[i:end] = rng.normal(-p["drift"] * 3, p["vol"], seg)
        elif regime == "range":
            returns[i:end] = rng.normal(0, p["vol"] * 0.5, seg)
        else:
            returns[i:end] = rng.normal(-p["drift"], p["vol"] * 2.0, seg)
        i = end

    close = p["base"] * np.exp(np.cumsum(returns))
    intrabar = np.abs(returns) + rng.exponential(0.002, n)
    high = close * (1 + intrabar * 0.6)
    low = close * (1 - intrabar * 0.6)
    open_ = close * (1 + rng.randn(n) * 0.002)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    volume = (500 + rng.exponential(200, n)) * (1.0 + 3.0 * np.abs(returns) / 0.01)

    freq = _freq_for_tf(tf)
    idx = pd.date_range("2023-01-01", periods=n, freq=freq)
    ohlcv = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)

    rng_f = np.random.RandomState(42 + sym_hash + 7)
    funding = pd.Series(rng_f.normal(0.0001, 0.0004, n), index=idx)

    return ohlcv, funding

def run_combo(symbol, strategy, leverage, timeframe):
    n = _n_bars_for_tf(timeframe)
    ohlcv, funding = _generate_data(n, symbol, timeframe)
    warmup = _warmup_for_n(n)

    cfg = CryptoBacktestConfig(
        symbol=symbol,
        timeframe=timeframe,
        strategy_mode=strategy,
        leverage=float(leverage),
        initial_capital=10_000.0,
        compression_window=min(120, n // 8),
        ic_window=min(120, n // 8),
        ic_horizon=min(20, n // 50),
        funding_window=min(200, n // 6),
        regime_config=CryptoRegimeConfig(
            warmup_bars=warmup,
            rolling_window=warmup,
            refit_every=max(50, warmup // 5),
            vol_window=min(96, warmup // 4),
            cooldown_bars=3,
            circuit_breaker_dd=-0.25,
        ),
    )

    svc = CryptoBacktestService()
    result = svc.run(ohlcv, cfg, funding_rates=funding)
    return result, ohlcv

def export_trades(result, symbol, strategy, leverage, timeframe):
    trades = result["trades"]
    slug = f"{symbol.replace('/', '_').replace(':', '')}_{strategy}_{leverage}x_{timeframe}"
    path = OUT_DIR / f"{slug}.csv"

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "timestamp", "symbol", "side", "units", "price", "notional",
            "fee", "slippage", "funding_paid", "pnl", "exit_reason",
            "leverage", "bar_idx",
        ])
        for t in trades:
            w.writerow([
                t.timestamp.isoformat() if hasattr(t.timestamp, "isoformat") else str(t.timestamp),
                t.symbol, t.side, f"{t.units:.8f}", f"{t.price:.4f}",
                f"{t.notional:.4f}", f"{t.fee:.4f}", f"{t.slippage:.4f}",
                f"{t.funding_paid:.6f}", f"{t.pnl:.4f}", t.exit_reason,
                t.leverage, t.bar_idx,
            ])
    return path

def verify_trades(result, config_capital=10000.0):
    """Verify trade-level data integrity."""
    trades = result["trades"]
    issues = []

    for i, t in enumerate(trades):
        if t.price <= 0:
            issues.append(f"Trade {i}: negative price {t.price}")
        if t.units <= 0:
            issues.append(f"Trade {i}: non-positive units {t.units}")
        if t.notional < 0:
            issues.append(f"Trade {i}: negative notional {t.notional}")
        if t.fee < 0:
            issues.append(f"Trade {i}: negative fee {t.fee}")
        valid_sides = ("long_entry", "long_exit", "short_entry", "short_exit", "grid_buy", "grid_sell")
        if t.side not in valid_sides:
            issues.append(f"Trade {i}: unexpected side {t.side}")

    pnl_trades = [t for t in trades if t.pnl != 0]
    if pnl_trades:
        pnls = [t.pnl for t in pnl_trades]
        computed_win_rate = sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0
        reported_win_rate = result["win_rate"]
        if abs(computed_win_rate - reported_win_rate) > 0.02:
            issues.append(f"Win rate mismatch: computed={computed_win_rate:.3f} vs reported={reported_win_rate:.3f}")

    eq = result["equity_curve"]
    if float(eq.min()) < 0:
        issues.append(f"Equity went negative: min={float(eq.min()):.2f}")
    if abs(float(eq.iloc[0]) - config_capital) > 1.0:
        issues.append(f"Equity[0]={float(eq.iloc[0]):.2f} != initial_capital={config_capital}")

    final_eq = result["final_equity"]
    eq_last = float(eq.iloc[-1])
    if abs(final_eq - eq_last) > 1.0:
        issues.append(f"final_equity={final_eq:.2f} != equity[-1]={eq_last:.2f}")

    ret_from_eq = (final_eq / config_capital - 1) * 100
    ret_reported = result["total_return_pct"]
    if abs(ret_from_eq - ret_reported) > 0.5:
        issues.append(f"Return mismatch: from_equity={ret_from_eq:.1f}% vs reported={ret_reported:.1f}%")

    return issues

def verify_regimes(result):
    """Verify HMM produced all 3 regimes."""
    regimes = result["regimes"]
    unique = set(regimes.unique())
    issues = []
    for r in ["TRENDING", "RANGING", "STRESS"]:
        if r not in unique:
            issues.append(f"Missing regime: {r}")
    vc = regimes.value_counts(normalize=True)
    for r, pct in vc.items():
        if pct > 0.80:
            issues.append(f"Regime {r} dominates at {pct:.0%}")
    return issues

def verify_scores(result):
    """Verify scores have no NaN and reasonable distribution."""
    scores = result["scores"]
    issues = []
    if scores.isna().any():
        issues.append(f"NaN in scores: {scores.isna().sum()} bars")
    if (scores.abs() > 100).any():
        issues.append(f"Scores exceed [-100, 100]: max={float(scores.abs().max()):.1f}")
    return issues

def main():
    total = len(SYMBOLS) * len(STRATEGIES) * len(LEVERAGES) * len(TIMEFRAMES)
    print(f"Running {total} combinations ({len(SYMBOLS)} symbols x {len(STRATEGIES)} strategies x {len(LEVERAGES)} leverages x {len(TIMEFRAMES)} timeframes)")
    print(f"Exporting to: {OUT_DIR}\n")

    all_results = []
    all_issues = []
    combo_idx = 0

    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            for strategy in STRATEGIES:
                for lev in LEVERAGES:
                    combo_idx += 1
                    label = f"{sym} {strategy:12s} {lev:2d}x {tf:3s}"
                    try:
                        result, ohlcv = run_combo(sym, strategy, lev, tf)
                        csv_path = export_trades(result, sym, strategy, lev, tf)

                        trade_issues = verify_trades(result)
                        regime_issues = verify_regimes(result)
                        score_issues = verify_scores(result)
                        all_combo_issues = trade_issues + regime_issues + score_issues

                        status = "PASS" if not all_combo_issues else "WARN"
                        n_trades = result["n_trades"]
                        ret = result["total_return_pct"]
                        sharpe = result["sharpe"]
                        wr = result["win_rate"] * 100
                        mdd = result["max_drawdown"] * 100
                        bh = result["baselines"]["buy_and_hold"]["total_return_pct"]

                        print(f"[{combo_idx:3d}/{total}] {label} | trades={n_trades:>5d} ret={ret:>8.1f}% sharpe={sharpe:>6.2f} WR={wr:>5.1f}% MDD={mdd:>6.1f}% B&H={bh:>7.1f}% [{status}]")

                        if all_combo_issues:
                            for iss in all_combo_issues:
                                print(f"         ISSUE: {iss}")
                            all_issues.append((label, all_combo_issues))

                        all_results.append({
                            "symbol": sym, "strategy": strategy, "leverage": lev,
                            "timeframe": tf, "n_trades": n_trades, "return_pct": round(ret, 2),
                            "sharpe": round(sharpe, 4), "win_rate": round(wr, 1),
                            "max_drawdown_pct": round(mdd, 1), "buy_hold_pct": round(bh, 1),
                            "final_equity": round(result["final_equity"], 2),
                            "total_fees": round(result["total_fees"], 2),
                            "n_issues": len(all_combo_issues),
                            "csv_file": str(csv_path.name),
                        })

                    except Exception as e:
                        print(f"[{combo_idx:3d}/{total}] {label} | ERROR: {e}")
                        all_issues.append((label, [str(e)]))
                        all_results.append({
                            "symbol": sym, "strategy": strategy, "leverage": lev,
                            "timeframe": tf, "error": str(e),
                        })

    summary_path = OUT_DIR / "verification_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"VERIFICATION COMPLETE")
    print(f"{'='*80}")
    n_pass = sum(1 for r in all_results if r.get("n_issues", -1) == 0)
    n_warn = sum(1 for r in all_results if r.get("n_issues", 0) > 0)
    n_err = sum(1 for r in all_results if "error" in r)
    print(f"  PASS: {n_pass}/{total}")
    print(f"  WARN: {n_warn}/{total}")
    print(f"  ERROR: {n_err}/{total}")
    print(f"\n  Trade CSVs: {OUT_DIR}")
    print(f"  Summary JSON: {summary_path}")

    if all_issues:
        print(f"\nISSUES ({len(all_issues)} combos):")
        for label, issues in all_issues:
            for iss in issues:
                print(f"  {label}: {iss}")

    valid = [r for r in all_results if "error" not in r]
    if valid:
        print(f"\nAGGREGATE STATS:")
        rets = [r["return_pct"] for r in valid]
        sharpes = [r["sharpe"] for r in valid]
        print(f"  Return: min={min(rets):.1f}% median={np.median(rets):.1f}% max={max(rets):.1f}%")
        print(f"  Sharpe: min={min(sharpes):.2f} median={np.median(sharpes):.2f} max={max(sharpes):.2f}")
        print(f"  Total trades across all combos: {sum(r['n_trades'] for r in valid)}")
        print(f"  Total fees paid: ${sum(r['total_fees'] for r in valid):.2f}")

    return 0 if n_err == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
