#!/usr/bin/env python3
"""
Compute Information Coefficient and decile forward-return test for the composite score.

For each horizon h in {5, 20, 60, 252}, computes Spearman(score_t, return_{t+1..t+h})
and decile mean forward returns with standard errors. Uses the same scoring as production
(backend TechnicalIndicators + ScoringEngine via BacktestEngine._compute_rolling_scores).

Usage:
  python scripts/compute_ic_deciles.py --assets AAPL,SPY --horizons 5,20,60,252 --out validation/debug/ic_results.json

Requires: Run from project root with backend on PYTHONPATH (e.g. PYTHONPATH=backend python scripts/compute_ic_deciles.py ...).
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
# Backend first so "indicators" resolves to backend/indicators.py, not project indicators/
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# After path setup
from validation.metrics import information_coefficient, forward_returns


def compute_decile_forward_returns(
    scores: pd.Series,
    prices: pd.Series,
    horizon: int,
) -> dict:
    """Bin bars by score decile; return mean forward return and std err per decile."""
    fwd = forward_returns(prices, horizon=horizon)
    common = scores.index.intersection(fwd.index)
    scores_a = scores.reindex(common).dropna()
    fwd_a = fwd.reindex(common).dropna()
    common = scores_a.index.intersection(fwd_a.index)
    scores_a = scores_a.loc[common]
    fwd_a = fwd_a.loc[common]
    if len(scores_a) < 20:
        return {"deciles": [], "horizon": horizon}
    # Decile bins: 1 = lowest 10%, 10 = highest 10%
    decile = pd.qcut(scores_a.rank(method="first"), 10, labels=False, duplicates="drop") + 1
    if decile is None or hasattr(decile, "isna") and decile.isna().all():
        return {"deciles": [], "horizon": horizon}
    out = []
    for q in range(1, 11):
        mask = decile == q
        if mask.sum() == 0:
            continue
        rets = fwd_a.loc[mask]
        mean_ret = float(rets.mean())
        n = len(rets)
        se = float(rets.std() / np.sqrt(n)) if n > 1 and rets.std() > 0 else 0.0
        out.append({"decile": int(q), "mean_forward_return": mean_ret, "std_error": se, "n_bars": n})
    return {"deciles": out, "horizon": horizon}


def main():
    ap = argparse.ArgumentParser(description="Compute IC and decile forward returns for composite score")
    ap.add_argument("--assets", type=str, default="AAPL,SPY", help="Comma-separated symbols")
    ap.add_argument("--horizons", type=str, default="5,20,60,252", help="Comma-separated forward horizons (bars)")
    ap.add_argument("--out", type=str, default="validation/debug/ic_results.json", help="Output JSON path")
    ap.add_argument("--period", type=str, default="2y", help="History period for price data (e.g. 2y, max)")
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.assets.split(",") if s.strip()]
    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    if not horizons:
        horizons = [5, 20, 60, 252]

    from backend.data_providers import PriceProvider
    from backend.backtest import BacktestEngine

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for symbol in symbols:
        print(f"Loading and scoring {symbol} ...", file=sys.stderr)
        df = PriceProvider.fetch_historical_data(symbol, period=args.period)
        if df is None or df.empty or len(df) < 250:
            print(f"  Skip {symbol}: insufficient data", file=sys.stderr)
            all_results[symbol] = {"error": "insufficient_data", "n_bars": len(df) if df is not None else 0}
            continue
        # Normalize column names (yfinance may return Title case)
        cols = {c: c.title() if isinstance(c, str) else c for c in df.columns}
        df = df.rename(columns=cols)
        if "Close" not in df.columns:
            print(f"  Skip {symbol}: no Close column", file=sys.stderr)
            all_results[symbol] = {"error": "no_close_column"}
            continue

        scores = BacktestEngine._compute_rolling_scores(df)
        prices = df["Close"]

        ic_by_h = {}
        deciles_by_h = {}
        for h in horizons:
            fwd = forward_returns(prices, horizon=h)
            ic = information_coefficient(scores, fwd)
            ic_by_h[f"ic_{h}"] = float(ic) if not np.isnan(ic) else None
            deciles_by_h[f"horizon_{h}"] = compute_decile_forward_returns(scores, prices, h)

        all_results[symbol] = {
            "n_bars": len(scores),
            "ic_by_horizon": ic_by_h,
            "decile_forward_returns": deciles_by_h,
        }
        print(f"  ICs: {ic_by_h}", file=sys.stderr)

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Results written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
