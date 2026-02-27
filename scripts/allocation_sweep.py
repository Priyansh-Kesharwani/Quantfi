#!/usr/bin/env python3
"""Wide parameter sweep for AllocationEngine with cached regime computation."""
import sys
import os
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)  # line-buffered

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
import numpy as np
import pandas as pd
import itertools
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtester.portfolio_simulator import (
    AllocationEngine, AllocationConfig, prepare_multi_asset_data,
    _compute_jump_regime, _compute_raw_indicators, _drawdown_circuit_breaker,
    RISK_ON,
)
import backtester.portfolio_simulator as bps

def run_sweep(syms, start, end, label=""):
    print(f"\n{'='*80}")
    print(f"SWEEP: {label} | Assets: {syms} | {start.year}-{end.year}")
    print(f"{'='*80}")

    t0 = time.time()
    date_index, assets = prepare_multi_asset_data(
        symbols=syms,
        start_date=start,
        end_date=end,
        asset_meta={s: {"asset_type": "equity", "currency": "USD"} for s in syms},
        scoring_mode="adaptive",
    )
    print(f"Data prep: {time.time()-t0:.1f}s, {len(date_index)} bars")

    ref_sym = list(assets.keys())[0]
    ref_close = assets[ref_sym].close
    raw_ind = _compute_raw_indicators(ref_close, assets[ref_sym].high, assets[ref_sym].low)

    t1 = time.time()
    cached_regime, cached_prob = _compute_jump_regime(
        ref_close, date_index, raw_ind,
        n_states=2, jump_penalty=25.0, window=504, refit_every=63,
    )
    cached_cb = _drawdown_circuit_breaker(ref_close, lookback=20, threshold=-0.15)
    print(f"Regime pre-compute: {time.time()-t1:.1f}s")

    orig_jump = bps._compute_jump_regime
    orig_cb = bps._drawdown_circuit_breaker
    bps._compute_jump_regime = lambda *a, **k: (cached_regime.copy(), cached_prob.copy())
    bps._drawdown_circuit_breaker = lambda *a, **k: cached_cb.copy()

    grid = {
        "theta": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
        "risk_on": [0.92, 0.95, 0.97, 1.0],
        "risk_off": [0.60, 0.70, 0.80, 0.85, 0.90],
        "freq": [42, 63, 126, 252],
        "delta": [0.03, 0.05, 0.08, 0.10],
    }

    total_raw = 1
    for v in grid.values():
        total_raw *= len(v)
    print(f"Grid: {total_raw} raw combos")

    results = []
    count = 0
    t2 = time.time()
    bh_sharpe = bh_dd = bh_ret = None

    for theta, risk_on, risk_off, freq, delta in itertools.product(
        grid["theta"], grid["risk_on"], grid["risk_off"],
        grid["freq"], grid["delta"],
    ):
        if risk_off >= risk_on:
            continue
        count += 1

        cfg = AllocationConfig(
            symbols=syms,
            start_date=start,
            end_date=end,
            initial_capital=100000,
            scoring_mode="adaptive",
            risk_on_equity_pct=risk_on,
            risk_off_equity_pct=risk_off,
            theta_tilt=theta,
            rebalance_freq_days=freq,
            min_rebalance_delta=delta,
            drawdown_circuit_threshold=-0.15,
        )
        try:
            engine = AllocationEngine(cfg)
            result = engine.run(date_index, assets)
        except Exception:
            continue

        sharpe = float(result["sharpe_ratio"])
        max_dd = float(result["max_drawdown_pct"])
        ret = float(result["total_return_pct"])
        calmar = float(result.get("calmar_ratio", 0))
        ec = result.get("equity_curve", [])
        inv = float(np.mean([e.get("invested_pct", 0) for e in ec])) if ec else 0
        trades = int(result["total_trades"])

        bh = result.get("benchmarks", {}).get("buy_and_hold", {})
        if bh_sharpe is None:
            bh_sharpe = float(bh.get("sharpe_ratio", 0))
            bh_dd = float(bh.get("max_drawdown_pct", -100))
            bh_ret = float(bh.get("total_return_pct", 0))

        if sharpe > bh_sharpe and max_dd > bh_dd and inv > 70:
            results.append(dict(
                theta=theta, risk_on=risk_on, risk_off=risk_off,
                freq=freq, delta=delta,
                sharpe=sharpe, ret=ret, dd=max_dd, calmar=calmar,
                inv=inv, trades=trades, composite=sharpe * calmar,
            ))

        if count % 200 == 0:
            elapsed = time.time() - t2
            rate = count / elapsed
            print(f"  {count} tested, {len(results)} winners, {rate:.0f} cfg/s")

    elapsed = time.time() - t2
    print(f"\nDone: {count} configs in {elapsed:.1f}s ({count/elapsed:.0f} cfg/s)")
    print(f"B&H: return={bh_ret:.1f}%, Sharpe={bh_sharpe:.3f}, DD={bh_dd:.1f}%")
    print(f"Winners (beat B&H on Sharpe AND MaxDD): {len(results)}")

    bps._compute_jump_regime = orig_jump
    bps._drawdown_circuit_breaker = orig_cb

    if not results:
        print("NO winning configurations found!")
        return None

    results.sort(key=lambda r: r["sharpe"], reverse=True)

    hdr = f"{'theta':>6} {'on':>5} {'off':>5} {'freq':>5} {'delta':>6} | {'sharpe':>7} {'ret%':>8} {'dd%':>7} {'calmar':>7} {'inv%':>5} {'trades':>6}"
    sep = "-" * 85

    print(f"\nTop 15 by Sharpe:")
    print(hdr); print(sep)
    for r in results[:15]:
        print(f"{r['theta']:>6.2f} {r['risk_on']:>5.2f} {r['risk_off']:>5.2f} {r['freq']:>5d} {r['delta']:>6.2f} | {r['sharpe']:>7.3f} {r['ret']:>7.1f}% {r['dd']:>6.1f}% {r['calmar']:>7.3f} {r['inv']:>5.1f} {r['trades']:>6d}")

    results.sort(key=lambda r: r["ret"], reverse=True)
    print(f"\nTop 15 by Return:")
    print(hdr); print(sep)
    for r in results[:15]:
        print(f"{r['theta']:>6.2f} {r['risk_on']:>5.2f} {r['risk_off']:>5.2f} {r['freq']:>5d} {r['delta']:>6.2f} | {r['sharpe']:>7.3f} {r['ret']:>7.1f}% {r['dd']:>6.1f}% {r['calmar']:>7.3f} {r['inv']:>5.1f} {r['trades']:>6d}")

    results.sort(key=lambda r: r["composite"], reverse=True)
    print(f"\nTop 15 by Sharpe*Calmar:")
    print(hdr); print(sep)
    for r in results[:15]:
        print(f"{r['theta']:>6.2f} {r['risk_on']:>5.2f} {r['risk_off']:>5.2f} {r['freq']:>5d} {r['delta']:>6.2f} | {r['sharpe']:>7.3f} {r['ret']:>7.1f}% {r['dd']:>6.1f}% {r['calmar']:>7.3f} {r['inv']:>5.1f} {r['trades']:>6d}")

    results.sort(key=lambda r: r["sharpe"], reverse=True)
    best = results[0]
    print(f"\nBEST ({label}): theta={best['theta']}, on={best['risk_on']}, off={best['risk_off']}, freq={best['freq']}, delta={best['delta']}")
    print(f"  Sharpe={best['sharpe']:.3f}, Return={best['ret']:.1f}%, DD={best['dd']:.1f}%, Calmar={best['calmar']:.3f}")

    return {
        "label": label,
        "symbols": syms,
        "best_by_sharpe": results[0],
        "best_by_composite": sorted(results, key=lambda r: r["composite"], reverse=True)[0],
        "best_by_return": sorted(results, key=lambda r: r["ret"], reverse=True)[0],
        "benchmark": {"bh_return_pct": bh_ret, "bh_sharpe": bh_sharpe, "bh_max_dd_pct": bh_dd},
        "total_tested": count,
        "winners": len(results),
        "top_20": results[:20],
    }


if __name__ == "__main__":
    all_results = {}

    r1 = run_sweep(
        ["SPY", "AAPL", "TLT", "GLD"],
        datetime(2015, 1, 1), datetime(2025, 1, 1),
        label="4-asset core"
    )
    if r1:
        all_results["4_asset_core"] = r1

    r2 = run_sweep(
        ["SPY", "AAPL", "MSFT", "TLT", "GLD"],
        datetime(2015, 1, 1), datetime(2025, 1, 1),
        label="5-asset +MSFT"
    )
    if r2:
        all_results["5_asset_msft"] = r2

    r3 = run_sweep(
        ["SPY", "AAPL", "MSFT", "AMZN", "TLT", "GLD"],
        datetime(2015, 1, 1), datetime(2025, 1, 1),
        label="6-asset mega"
    )
    if r3:
        all_results["6_asset_mega"] = r3

    r4 = run_sweep(
        ["SPY", "AAPL", "TLT", "GLD"],
        datetime(2018, 1, 1), datetime(2025, 1, 1),
        label="4-asset recent (2018-2025)"
    )
    if r4:
        all_results["4_asset_recent"] = r4

    print("\n" + "=" * 80)
    print("SUMMARY: Best config per universe")
    print("=" * 80)
    for k, v in all_results.items():
        b = v["best_by_sharpe"]
        bm = v["benchmark"]
        print(f"\n{v['label']} ({v['winners']}/{v['total_tested']} winners):")
        print(f"  Best: theta={b['theta']}, on={b['risk_on']}, off={b['risk_off']}, freq={b['freq']}, delta={b['delta']}")
        print(f"  Strategy: Sharpe={b['sharpe']:.3f}, Return={b['ret']:.1f}%, DD={b['dd']:.1f}%")
        print(f"  B&H:      Sharpe={bm['bh_sharpe']:.3f}, Return={bm['bh_return_pct']:.1f}%, DD={bm['bh_max_dd_pct']:.1f}%")
        print(f"  Improvement: Sharpe +{(b['sharpe']-bm['bh_sharpe'])/bm['bh_sharpe']*100:.1f}%, DD +{(b['dd']-bm['bh_max_dd_pct'])/abs(bm['bh_max_dd_pct'])*100:.1f}%")

    out_path = os.path.join(os.path.dirname(__file__), "..", "validation", "outputs", "allocation_sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")
