#!/usr/bin/env python3
# RESEARCH ONLY — DO NOT USE OUTPUT FOR STRATEGY PARAMETERS
"""
Regime clustering and regime-sliced IC/deciles.

Clusters bars by rolling volatility and trend proxy (price vs SMA50), then computes
IC and decile forward returns per regime. Output: validation/debug/regime_ic_deciles.json.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from validation.metrics import information_coefficient, forward_returns


def compute_decile_forward_returns(scores: pd.Series, prices: pd.Series, horizon: int) -> dict:
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
    decile = pd.qcut(scores_a.rank(method="first"), 10, labels=False, duplicates="drop") + 1
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
    ap = argparse.ArgumentParser(description="Regime clustering + IC/deciles per regime")
    ap.add_argument("--assets", type=str, default="AAPL,SPY", help="Comma-separated symbols")
    ap.add_argument("--horizons", type=str, default="5,20,60", help="Comma-separated horizons")
    ap.add_argument("--n-clusters", type=int, default=3, help="Number of regimes (k-means)")
    ap.add_argument("--vol-window", type=int, default=21, help="Rolling window for volatility")
    ap.add_argument("--out", type=str, default="validation/debug/regime_ic_deciles.json", help="Output JSON path")
    ap.add_argument("--period", type=str, default="2y", help="History period")
    args = ap.parse_args()

    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print("sklearn required: pip install scikit-learn", file=sys.stderr)
        return 1

    symbols = [s.strip().upper() for s in args.assets.split(",") if s.strip()]
    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    if not horizons:
        horizons = [5, 20, 60]

    from backend.data_providers import PriceProvider
    from backend.backtest import BacktestEngine

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for symbol in symbols:
        print(f"Loading and scoring {symbol} ...", file=sys.stderr)
        df = PriceProvider.fetch_historical_data(symbol, period=args.period)
        if df is None or df.empty or len(df) < 300:
            all_results[symbol] = {"error": "insufficient_data"}
            continue
        cols = {c: c.title() if isinstance(c, str) else c for c in df.columns}
        df = df.rename(columns=cols)
        if "Close" not in df.columns:
            all_results[symbol] = {"error": "no_close"}
            continue

        scores = BacktestEngine._compute_rolling_scores(df)
        prices = df["Close"]
        returns = prices.pct_change().dropna()
        vol = returns.rolling(args.vol_window, min_periods=5).std()
        sma50 = prices.rolling(50, min_periods=1).mean()
        trend = (prices / sma50 - 1.0).replace([np.inf, -np.inf], np.nan)

        # Align and dropna for clustering
        common = scores.index.intersection(vol.index).intersection(trend.index)
        vol_a = vol.reindex(common).ffill().bfill().fillna(0)
        trend_a = trend.reindex(common).ffill().bfill().fillna(0)
        scores_a = scores.reindex(common).dropna()
        valid = vol_a.notna() & trend_a.notna() & scores_a.notna()
        valid = valid & (vol_a > 0)  # avoid zero vol
        if valid.sum() < 100:
            all_results[symbol] = {"error": "insufficient_valid_bars"}
            continue

        X = np.column_stack([
            vol_a.loc[valid].values,
            trend_a.loc[valid].values,
        ])
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
        kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        regime_series = pd.Series(np.nan, index=common)
        regime_series.loc[valid] = labels

        all_results[symbol] = {"n_bars": int(valid.sum()), "n_clusters": args.n_clusters}

        for r in range(args.n_clusters):
            mask = regime_series == r
            if mask.sum() < 50:
                all_results[symbol][f"regime_{r}"] = {"error": "too_few_bars", "n_bars": int(mask.sum())}
                continue
            s_r = scores_a.loc[mask]
            p_r = prices.reindex(s_r.index).ffill().bfill()
            ic_by_h = {}
            deciles_by_h = {}
            for h in horizons:
                fwd = forward_returns(p_r, horizon=h)
                ic_by_h[f"ic_{h}"] = float(information_coefficient(s_r, fwd)) if not (np.isnan(information_coefficient(s_r, fwd))) else None
                deciles_by_h[f"horizon_{h}"] = compute_decile_forward_returns(s_r, p_r, h)
            all_results[symbol][f"regime_{r}"] = {
                "n_bars": int(mask.sum()),
                "ic_by_horizon": ic_by_h,
                "decile_forward_returns": deciles_by_h,
            }
        print(f"  regimes: {args.n_clusters}, bars per regime: {[all_results[symbol].get(f'regime_{r}', {}).get('n_bars') for r in range(args.n_clusters)]}", file=sys.stderr)

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
