#!/usr/bin/env python3
"""
Suggest score component weights from rolling IC of each sub-score vs forward returns.

Loads OHLCV, computes per-bar composite and component scores (S_T, S_V, S_S, S_M) via
backend ScoringEngine, then for a rolling window computes IC of each component vs
forward return and suggests weights (e.g. proportional to max(0, IC) normalized).
Output: validation/debug/rolling_ic_weights.json
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

def compute_scores_and_breakdown(df, backend_engine):
    """Return (scores, breakdown_df) with columns technical_momentum, volatility_opportunity, statistical_deviation, macro_fx."""
    BacktestEngine = backend_engine
    n = len(df)
    scores = pd.Series(50.0, index=df.index, dtype=float)
    breakdown = {k: np.full(n, np.nan) for k in ["technical_momentum", "volatility_opportunity", "statistical_deviation", "macro_fx"]}
    if n < 200:
        return scores, pd.DataFrame(breakdown, index=df.index)
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    from backend.indicators import TechnicalIndicators
    from backend.scoring import ScoringEngine
    from backend.data_providers import FXProvider
    sma_50 = TechnicalIndicators.calculate_sma(close, 50)
    sma_200 = TechnicalIndicators.calculate_sma(close, 200)
    rsi_14 = TechnicalIndicators.calculate_rsi(close, 14)
    macd_data = TechnicalIndicators.calculate_macd(close)
    macd_line, macd_hist = macd_data["macd"], macd_data["histogram"]
    bb = TechnicalIndicators.calculate_bollinger_bands(close)
    atr_14 = TechnicalIndicators.calculate_atr(high, low, close, 14)
    atr_pctl = TechnicalIndicators.calculate_atr_percentile(atr_14, min(252, max(30, n // 4)))
    z20 = TechnicalIndicators.calculate_z_score(close, 20)
    z50 = TechnicalIndicators.calculate_z_score(close, 50)
    z100 = TechnicalIndicators.calculate_z_score(close, 100)
    drawdown = TechnicalIndicators.calculate_drawdown(close)
    adx_14 = TechnicalIndicators.calculate_adx(high, low, close, 14)
    usd_inr_rate = FXProvider.fetch_usd_inr_rate() or 83.5

    def _safe(series, idx):
        v = series.iloc[idx]
        return float(v) if pd.notna(v) else None

    for i in range(n):
        ind = {
            "sma_200": _safe(sma_200, i),
            "sma_50": _safe(sma_50, i),
            "rsi_14": _safe(rsi_14, i),
            "macd": _safe(macd_line, i),
            "macd_hist": _safe(macd_hist, i),
            "bb_lower": _safe(bb["lower"], i),
            "bb_upper": _safe(bb["upper"], i),
            "atr_percentile": _safe(atr_pctl, i),
            "drawdown_pct": _safe(drawdown, i),
            "z_score_20": _safe(z20, i),
            "z_score_50": _safe(z50, i),
            "z_score_100": _safe(z100, i),
            "adx_14": _safe(adx_14, i),
        }
        current_price = float(close.iloc[i])
        if pd.isna(current_price):
            continue
        composite, bd, _ = ScoringEngine.calculate_composite_score(ind, current_price, usd_inr_rate)
        scores.iloc[i] = composite
        breakdown["technical_momentum"][i] = bd.technical_momentum
        breakdown["volatility_opportunity"][i] = bd.volatility_opportunity
        breakdown["statistical_deviation"][i] = bd.statistical_deviation
        breakdown["macro_fx"][i] = bd.macro_fx

    return scores, pd.DataFrame(breakdown, index=df.index)

def main():
    ap = argparse.ArgumentParser(description="Suggest score weights from rolling IC per component")
    ap.add_argument("--assets", type=str, default="AAPL,SPY")
    ap.add_argument("--horizon", type=int, default=20, help="Forward return horizon")
    ap.add_argument("--rolling-window", type=int, default=252)
    ap.add_argument("--out", type=str, default="validation/debug/rolling_ic_weights.json")
    ap.add_argument("--period", type=str, default="2y")
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.assets.split(",") if s.strip()]
    from backend.data_providers import PriceProvider
    from backend.backtest import BacktestEngine

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for symbol in symbols:
        print(f"Loading {symbol} ...", file=sys.stderr)
        df = PriceProvider.fetch_historical_data(symbol, period=args.period)
        if df is None or df.empty or len(df) < 300:
            all_results[symbol] = {"error": "insufficient_data"}
            continue
        cols = {c: c.title() if isinstance(c, str) else c for c in df.columns}
        df = df.rename(columns=cols)
        if "Close" not in df.columns:
            all_results[symbol] = {"error": "no_close"}
            continue

        scores, breakdown_df = compute_scores_and_breakdown(df, BacktestEngine)
        prices = df["Close"]
        fwd = forward_returns(prices, horizon=args.horizon)
        components = ["technical_momentum", "volatility_opportunity", "statistical_deviation", "macro_fx"]
        rolling_ic = {c: [] for c in components}
        n = len(scores)
        w = args.rolling_window
        for start in range(0, n - w - args.horizon):
            end = start + w
            fwd_slice = fwd.iloc[start:end]
            valid = fwd_slice.notna()
            if valid.sum() < 30:
                continue
            ics = {}
            for c in components:
                ser = breakdown_df[c].iloc[start:end]
                common = ser.dropna().index.intersection(fwd_slice[valid].index)
                if len(common) < 30:
                    ics[c] = np.nan
                else:
                    ics[c] = information_coefficient(ser.loc[common], fwd_slice.loc[common])
            for c in components:
                rolling_ic[c].append(ics.get(c, np.nan))
        suggested = {}
        last_ics = {c: (np.nanmean([x for x in rolling_ic[c] if not np.isnan(x)]) if rolling_ic[c] else np.nan) for c in components}
        total = sum(max(0, last_ics[c]) for c in components if not np.isnan(last_ics[c]))
        if total > 0:
            for c in components:
                suggested[c] = round(max(0, last_ics.get(c, 0)) / total, 4)
        else:
            suggested = {c: 0.25 for c in components}
        all_results[symbol] = {
            "rolling_ic_mean": last_ics,
            "suggested_weights": suggested,
            "n_windows": len(rolling_ic["technical_momentum"]),
        }
        print(f"  ICs: {last_ics}", file=sys.stderr)
        print(f"  Suggested weights: {suggested}", file=sys.stderr)

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results written to {out_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
