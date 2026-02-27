#!/usr/bin/env python3
"""
Create a deterministic refactor verification snapshot.

Fetches OHLCV for given symbols and date range, canonicalizes column names,
and writes a single Parquet to data_snapshots/refactor_verification.parquet
for use in deterministic unit tests and verify_refactor_determinism.py.

Usage:
    python scripts/create_refactor_snapshot.py --symbols SPY AAPL --start 2020-01-01 --end 2023-12-31
    python scripts/create_refactor_snapshot.py
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yfinance as yf

def fetch_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch daily OHLCV from yfinance; return DataFrame with lowercase columns."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, auto_adjust=True)
        if df is None or df.empty:
            return None
        df.columns = [c.lower() for c in df.columns]
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(df.columns)):
            return None
        df = df[list(required)]
        df["symbol"] = symbol
        return df
    except Exception:
        return None

def main() -> int:
    parser = argparse.ArgumentParser(description="Create refactor verification snapshot")
    parser.add_argument("--symbols", nargs="+", default=["SPY", "AAPL"])
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2023-12-31")
    parser.add_argument("--output", default=None, help="Override output path")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "data_snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else out_dir / "refactor_verification.parquet"

    frames = []
    for sym in args.symbols:
        df = fetch_ohlcv(sym, args.start, args.end)
        if df is not None:
            frames.append(df)
        else:
            print(f"Warning: no data for {sym}", file=sys.stderr)

    if not frames:
        print("Error: no data fetched for any symbol", file=sys.stderr)
        return 1

    combined = pd.concat(frames, axis=0)
    combined = combined.sort_index()
    combined.to_parquet(out_path, index=True)
    print(f"Wrote {len(combined)} rows to {out_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
