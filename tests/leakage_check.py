"""
Leakage check: compare full-dataset indicator run vs incremental past-only run.

Run A: Compute indicators on the full dataset in one shot (same entry points as
       backtest/simulator).
Run B: For each timestamp t, run the same pipeline on df.iloc[:t+1] and record
       the value at the last index only (no access to future rows).
If at any t the two values differ beyond a small tolerance, count as mismatch
(leakage detected).

Usage:
  python -m tests.leakage_check
  pytest tests/leakage_check.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.indicators import TechnicalIndicators
from backend.app_config import get_backend_config

CFG = get_backend_config()
TOL = 1e-9
MIN_HISTORY = getattr(CFG, "indicator_min_history_rows", 200)


def _make_fixture_df(n: int = 260, seed: int = 42) -> pd.DataFrame:
    """Small deterministic OHLCV DataFrame with backend column names."""
    from tests.fixtures import fbm_series
    df = fbm_series(n=n, seed=seed)
    return df.rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close",
    })[["Open", "High", "Low", "Close"]]


def _run_a_full_series(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Run A: compute each indicator as a full series on the full dataset."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    z0, z1, z2 = CFG.z_score_periods

    sma_50 = TechnicalIndicators.calculate_sma(close, CFG.sma_short_period)
    sma_200 = TechnicalIndicators.calculate_sma(close, CFG.sma_long_period)
    ema_50 = TechnicalIndicators.calculate_ema(close, CFG.ema_period)
    rsi_14 = TechnicalIndicators.calculate_rsi(close)
    macd_data = TechnicalIndicators.calculate_macd(close)
    bb = TechnicalIndicators.calculate_bollinger_bands(close)
    atr_series = TechnicalIndicators.calculate_atr(high, low, close)
    atr_percentile = TechnicalIndicators.calculate_atr_percentile(atr_series)
    z_score_20 = TechnicalIndicators.calculate_z_score(close, z0)
    z_score_50 = TechnicalIndicators.calculate_z_score(close, z1)
    z_score_100 = TechnicalIndicators.calculate_z_score(close, z2)
    drawdown_pct = TechnicalIndicators.calculate_drawdown(close)
    adx_14 = TechnicalIndicators.calculate_adx(high, low, close)

    return {
        "sma_50": sma_50,
        "sma_200": sma_200,
        "ema_50": ema_50,
        "rsi_14": rsi_14,
        "macd": macd_data["macd"],
        "macd_signal": macd_data["signal"],
        "macd_hist": macd_data["histogram"],
        "bb_upper": bb["upper"],
        "bb_middle": bb["middle"],
        "bb_lower": bb["lower"],
        "atr_14": atr_series,
        "atr_percentile": atr_percentile,
        "z_score_20": z_score_20,
        "z_score_50": z_score_50,
        "z_score_100": z_score_100,
        "drawdown_pct": drawdown_pct,
        "adx_14": adx_14,
    }


def _run_b_incremental(df: pd.DataFrame) -> list[dict]:
    """Run B: for each t, run pipeline on df.iloc[:t+1], record last value only."""
    n = len(df)
    out: list[dict] = []
    for t in range(n):
        slice_df = df.iloc[: t + 1]
        row = TechnicalIndicators.calculate_all_indicators(slice_df)
        out.append(row)
    return out


def _compare_at_t(
    run_a_series: dict[str, pd.Series],
    run_b_row: dict,
    t: int,
    tolerance: float = TOL,
) -> dict[str, bool]:
    result = {}
    for name, series in run_a_series.items():
        a_val = series.iloc[t] if t < len(series) else np.nan
        a_nan = pd.isna(a_val)
        b_val = run_b_row.get(name)
        b_nan = b_val is None or (isinstance(b_val, float) and np.isnan(b_val))
        if a_nan and b_nan:
            result[name] = True
        elif a_nan or b_nan:
            result[name] = False
        else:
            a_f = float(a_val)
            b_float = float(b_val)
            if np.isinf(a_f) and np.isinf(b_float):
                result[name] = bool(np.sign(a_f) == np.sign(b_float))
            else:
                result[name] = bool(np.abs(a_f - b_float) <= tolerance)
    return result


def run_leakage_check(
    df: pd.DataFrame | None = None,
    tolerance: float = TOL,
    min_history: int | None = None,
) -> dict[str, dict]:
    """
    Run full vs incremental comparison. Returns per-indicator summary.

    Returns
    -------
    dict[str, dict]
        indicator_name -> {"mismatch_count": int, "first_mismatch_index": int | None}
    """
    if df is None:
        df = _make_fixture_df()
    min_hist = min_history if min_history is not None else MIN_HISTORY
    n = len(df)

    run_a_series = _run_a_full_series(df)
    run_b_rows = _run_b_incremental(df)

    summary: dict[str, dict] = {}
    for name in run_a_series:
        mismatch_count = 0
        first_mismatch_index: int | None = None
        for t in range(n):
            run_b_row = run_b_rows[t]
            if not run_b_row:
                # Backend returns {} when len(df) < min_history; skip comparison
                continue
            matches = _compare_at_t(run_a_series, run_b_row, t, tolerance)
            if not matches.get(name, True):
                mismatch_count += 1
                if first_mismatch_index is None:
                    first_mismatch_index = t
        summary[name] = {
            "mismatch_count": mismatch_count,
            "first_mismatch_index": first_mismatch_index,
        }
    return summary


def print_summary_table(summary: dict[str, dict]) -> None:
    """Print: Indicator Name | mismatch count | first mismatch index."""
    print("\nIndicator Name           | mismatch count | first mismatch index")
    print("-" * 65)
    for name, data in sorted(summary.items()):
        mc = data["mismatch_count"]
        fi = data["first_mismatch_index"]
        fi_str = str(fi) if fi is not None else "—"
        print(f"{name:24} | {mc:14} | {fi_str}")
    any_leak = any(d["mismatch_count"] > 0 for d in summary.values())
    if any_leak:
        print("\n*** LEAKAGE DETECTED: at least one indicator has full vs incremental mismatch.")
    else:
        print("\nNo leakage detected (full and incremental runs match within tolerance).")


def main() -> int:
    df = _make_fixture_df()
    summary = run_leakage_check(df=df, tolerance=TOL)
    print_summary_table(summary)
    return 1 if any(d["mismatch_count"] > 0 for d in summary.values()) else 0


def test_leakage_check_runs_and_prints_summary(capsys):
    """Run full vs incremental comparison; summary has expected structure."""
    df = _make_fixture_df(n=260)
    summary = run_leakage_check(df=df, tolerance=TOL)
    assert "sma_50" in summary
    assert "z_score_20" in summary
    for name, data in summary.items():
        assert "mismatch_count" in data
        assert "first_mismatch_index" in data
    # Optional: print table when run with -s
    print_summary_table(summary)
    out, _ = capsys.readouterr()
    assert "mismatch count" in out
    assert "first mismatch index" in out


if __name__ == "__main__":
    sys.exit(main())
