"""
Refactor path — IC and decile forward-return report for CompositeScore.

For each asset and horizon in [5, 20, 60, 252], computes Spearman IC and
decile mean forward returns. Writes validation/reports/ic_refactor.json.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from validation.metrics import information_coefficient, forward_returns

def _refactor_composite_score_from_df(df: pd.DataFrame) -> pd.Series:
    """Build refactor components and return Entry CompositeScore series."""
    from indicators.refactor_components import ofi_refactor, vwap_z_refactor, hurst_refactor
    from indicators.composite_refactor import compute_composite_score_refactor
    from indicators.normalization_refactor import canonical_normalize

    cols = {c: c.lower() if isinstance(c, str) else c for c in df.columns}
    df = df.rename(columns=cols)
    close = df["close"].values
    volume = df["volume"].values if "volume" in df.columns else np.ones(len(df))

    ofi_s, _ = ofi_refactor(df, window=20, min_obs=50)
    vwap_s, _ = vwap_z_refactor(close, volume, vol_window=60, index=df.index)
    vwap_norm, _ = canonical_normalize(np.nan_to_num(vwap_s.values, nan=0), mode="approx", higher_is_favorable=False, min_obs=50)
    vwap_norm_s = pd.Series(vwap_norm, index=df.index)
    hurst_s, _ = hurst_refactor(close, window=min(200, len(df) // 2), method="rs", index=df.index)
    hurst_norm, _ = canonical_normalize(np.nan_to_num(hurst_s.values, nan=0.5), mode="approx", min_obs=50)
    hurst_norm_s = pd.Series(hurst_norm, index=df.index)

    n = len(df)
    idx = df.index
    components = {
        "T_t": pd.Series(0.5 * np.ones(n), index=idx),
        "U_t": vwap_norm_s.reindex(idx).fillna(0.5),
        "H_t": hurst_norm_s.reindex(idx).fillna(0.5),
        "LDC_t": pd.Series(0.5 * np.ones(n), index=idx),
        "O_t": ofi_s.reindex(idx).fillna(0.5),
        "C_t": pd.Series(0.9 * np.ones(n), index=idx),
        "L_t": pd.Series(0.9 * np.ones(n), index=idx),
        "R_t": pd.Series(0.5 * np.ones(n), index=idx),
        "TBL_flag": pd.Series(0.5 * np.ones(n), index=idx),
        "OFI_rev": pd.Series(0.5 * np.ones(n), index=idx),
        "lambda_decay": pd.Series(0.5 * np.ones(n), index=idx),
    }
    entry, _, _ = compute_composite_score_refactor(components)
    return entry

def compute_decile_forward_returns(scores: pd.Series, prices: pd.Series, horizon: int) -> dict:
    """Decile mean forward return and std err per decile."""
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

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Refactor IC and decile report")
    parser.add_argument("--snapshot", type=str, default=None, help="Path to refactor snapshot parquet")
    parser.add_argument("--out", type=str, default="validation/reports/ic_refactor.json")
    parser.add_argument("--horizons", type=str, default="5,20,60,252")
    args = parser.parse_args()

    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    if not horizons:
        horizons = [5, 20, 60, 252]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.snapshot:
        path = Path(args.snapshot)
        if not path.exists():
            path = PROJECT_ROOT / path
        df = pd.read_parquet(path)
        if "symbol" in df.columns:
            symbols = df["symbol"].unique().tolist()
        else:
            symbols = ["UNKNOWN"]
    else:
        snapshot_path = PROJECT_ROOT / "data_snapshots" / "refactor_verification.parquet"
        if not snapshot_path.exists():
            print("No snapshot; run scripts/create_refactor_snapshot.py first or pass --snapshot", file=sys.stderr)
            return 1
        df = pd.read_parquet(snapshot_path)
        symbols = df["symbol"].unique().tolist() if "symbol" in df.columns else ["UNKNOWN"]

    all_results = {}
    for sym in symbols:
        if "symbol" in df.columns:
            sub = df.loc[df["symbol"] == sym].copy()
        else:
            sub = df.copy()
        if len(sub) < 100:
            all_results[sym] = {"error": "insufficient_bars", "n_bars": len(sub)}
            continue
        try:
            scores = _refactor_composite_score_from_df(sub)
            prices = sub["close"] if "close" in sub.columns else sub["Close"]
            ic_by_h = {}
            deciles_by_h = {}
            for h in horizons:
                fwd = forward_returns(prices, horizon=h)
                ic = information_coefficient(scores, fwd)
                ic_by_h[f"ic_{h}"] = float(ic) if not np.isnan(ic) else None
                deciles_by_h[f"horizon_{h}"] = compute_decile_forward_returns(scores, prices, h)
            all_results[sym] = {"n_bars": len(scores), "ic_by_horizon": ic_by_h, "decile_forward_returns": deciles_by_h}
        except Exception as e:
            all_results[sym] = {"error": str(e), "n_bars": len(sub)}

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Wrote {out_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
