#!/usr/bin/env python3
"""
IC Diagnostic: compute rank Information Coefficient for each indicator
against forward N-day returns to validate predictive power.

Usage:
    python scripts/ic_diagnostic.py [--symbols SPY AAPL] [--start 2018-01-01] [--end 2024-01-01]

Output: validation/outputs/ic_diagnostic.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _compute_raw_indicators(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute all indicator arrays from OHLC data."""
    n = len(close)
    s_close = pd.Series(close)
    s_high = pd.Series(high)
    s_low = pd.Series(low)

    sma50 = s_close.rolling(50).mean().values
    sma200 = s_close.rolling(200).mean().values

    delta = s_close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean().values
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(loss != 0, gain / loss, 0)
    rsi = 100 - (100 / (1 + rs))

    ema_f = s_close.ewm(span=12, adjust=False).mean().values
    ema_s = s_close.ewm(span=26, adjust=False).mean().values
    macd_line = ema_f - ema_s
    macd_sig = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values
    macd_hist = macd_line - macd_sig

    bb_mid = s_close.rolling(20).mean().values
    bb_std = s_close.rolling(20).std().values
    bb_lower = bb_mid - 2 * bb_std
    with np.errstate(divide="ignore", invalid="ignore"):
        bb_pctb = np.where(
            (bb_mid + 2 * bb_std - bb_lower) != 0,
            (close - bb_lower) / (bb_mid + 2 * bb_std - bb_lower),
            0.5,
        )

    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0], tr3[0] = tr1[0], tr1[0]
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_arr = pd.Series(true_range).rolling(14).mean().values
    atr_pctl = pd.Series(atr_arr).rolling(min(252, max(30, n // 4))).apply(
        lambda x: pd.Series(x).rank().iloc[-1] / len(x) * 100, raw=False
    ).values

    zs = []
    for zp in (20, 50, 100):
        rm = s_close.rolling(zp).mean().values
        rs_ = s_close.rolling(zp).std().values
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(rs_ != 0, (close - rm) / rs_, 0)
        zs.append(z)
    avg_z = np.nanmean(zs, axis=0)
    avg_z = np.where(np.isnan(avg_z), 0.0, avg_z)

    running_max = np.maximum.accumulate(close)
    drawdown = np.where(running_max > 0, (close - running_max) / running_max * 100, 0)

    high_diff = s_high.diff().values
    low_diff = (-s_low.diff()).values
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    atr_adx = pd.Series(true_range).rolling(14).mean().values
    plus_di = np.where(atr_adx != 0, 100 * pd.Series(plus_dm).rolling(14).mean().values / atr_adx, 0)
    minus_di = np.where(atr_adx != 0, 100 * pd.Series(minus_dm).rolling(14).mean().values / atr_adx, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dx = np.where(
            (plus_di + minus_di) != 0,
            100 * np.abs(plus_di - minus_di) / (plus_di + minus_di),
            0,
        )
    adx = pd.Series(dx).rolling(14).mean().values

    sma200_dist = np.where(sma200 != 0, (close - sma200) / sma200 * 100, 0.0)

    return {
        "rsi": rsi,
        "adx": adx,
        "macd_line": macd_line,
        "macd_hist": macd_hist,
        "bb_pctb": bb_pctb,
        "atr_pctl": atr_pctl,
        "avg_z": avg_z,
        "drawdown": drawdown,
        "sma200_dist": sma200_dist,
    }


def compute_ic(
    indicator: np.ndarray,
    forward_returns: np.ndarray,
) -> Tuple[float, float, int]:
    """Rank IC (Spearman) with t-stat. Returns (ic, t_stat, n_valid)."""
    mask = ~(np.isnan(indicator) | np.isnan(forward_returns))
    x, y = indicator[mask], forward_returns[mask]
    n = len(x)
    if n < 30:
        return 0.0, 0.0, n
    ic, _ = spearmanr(x, y)
    t_stat = ic * np.sqrt((n - 2) / (1 - ic**2 + 1e-12))
    return float(ic), float(t_stat), n


def run_diagnostic(
    symbols: List[str],
    start_date: str,
    end_date: str,
) -> Dict:
    from backend.data_providers import PriceProvider

    lags = [5, 10, 20, 40]
    all_results = {}

    for sym in symbols:
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        fetch_start = start_dt - pd.Timedelta(days=400)

        df = PriceProvider.fetch_historical_data(
            sym, period="max",
            start_date=fetch_start,
            end_date=end_dt + pd.Timedelta(days=1),
        )
        if df is None or df.empty:
            print(f"  No data for {sym}, skipping")
            continue

        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        cl = df["Close"].values.astype(float)
        hi = df["High"].values.astype(float)
        lo = df["Low"].values.astype(float)

        indicators = _compute_raw_indicators(cl, hi, lo)

        fwd_rets = {}
        for lag in lags:
            fwd = np.full_like(cl, np.nan)
            fwd[:-lag] = (cl[lag:] - cl[:-lag]) / cl[:-lag]
            fwd_rets[lag] = fwd

        sym_results = {}
        for ind_name, ind_vals in indicators.items():
            row = {}
            for lag in lags:
                ic, t_stat, n = compute_ic(ind_vals, fwd_rets[lag])
                row[f"IC_{lag}d"] = round(ic, 4)
                row[f"tstat_{lag}d"] = round(t_stat, 2)
                row[f"n_{lag}d"] = n
            ic_20 = row.get("IC_20d", 0)
            t_20 = row.get("tstat_20d", 0)
            if abs(ic_20) >= 0.03 and abs(t_20) >= 2.0:
                row["verdict"] = "STRONG"
            elif abs(ic_20) >= 0.02:
                row["verdict"] = "WEAK"
            else:
                row["verdict"] = "NOISE"
            sym_results[ind_name] = row
        all_results[sym] = sym_results

    avg_results = {}
    if all_results:
        all_ind_names = list(next(iter(all_results.values())).keys())
        for ind_name in all_ind_names:
            avg_row = {}
            for lag in lags:
                ics = [all_results[s][ind_name][f"IC_{lag}d"] for s in all_results if ind_name in all_results[s]]
                avg_row[f"IC_{lag}d"] = round(np.mean(ics), 4) if ics else 0.0
            ic_20 = avg_row.get("IC_20d", 0)
            if abs(ic_20) >= 0.03:
                avg_row["verdict"] = "STRONG"
            elif abs(ic_20) >= 0.02:
                avg_row["verdict"] = "WEAK"
            else:
                avg_row["verdict"] = "NOISE"
            avg_results[ind_name] = avg_row

    return {"per_symbol": all_results, "average": avg_results}


def main():
    parser = argparse.ArgumentParser(description="IC Diagnostic for scoring indicators")
    parser.add_argument("--symbols", nargs="+", default=["SPY", "AAPL"])
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    args = parser.parse_args()

    print(f"Running IC diagnostic for {args.symbols} ({args.start} -> {args.end})")
    results = run_diagnostic(args.symbols, args.start, args.end)

    out_dir = PROJECT_ROOT / "validation" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ic_diagnostic.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\n{'='*60}")
    print("AVERAGE IC ACROSS ALL SYMBOLS")
    print(f"{'='*60}")
    print(f"{'Indicator':<16} {'IC_5d':>8} {'IC_10d':>8} {'IC_20d':>8} {'IC_40d':>8} {'Verdict':>8}")
    print("-" * 64)
    for ind, row in results.get("average", {}).items():
        print(
            f"{ind:<16} {row.get('IC_5d', 0):>8.4f} {row.get('IC_10d', 0):>8.4f} "
            f"{row.get('IC_20d', 0):>8.4f} {row.get('IC_40d', 0):>8.4f} {row.get('verdict', '?'):>8}"
        )


if __name__ == "__main__":
    main()
