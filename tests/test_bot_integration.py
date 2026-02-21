"""
Integration test: features -> scoring -> execution on a small canonical snapshot.

Writes artifacts to validation/artifacts/<runid>/; asserts deterministic outputs.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RUNID = "bot_integration_test"
ARTIFACTS_DIR = PROJECT_ROOT / "validation" / "artifacts" / RUNID


def _canonical_bar_snapshot(n: int = 80, seed: int = 42) -> pd.DataFrame:
    """Small deterministic bar DataFrame (open, high, low, close, volume)."""
    np.random.seed(seed)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n)) * 0.3
    low = close - np.abs(np.random.randn(n)) * 0.3
    open_ = np.roll(close, 1)
    open_[0] = 100
    volume = np.abs(np.random.randn(n) * 1e6 + 5e6)
    idx = pd.date_range("2020-01-01", periods=n, freq="min", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_integration_features_scoring_execution_deterministic():
    """
    Run features -> scoring -> execution with fixed seed; two runs produce same artifact hash.
    """
    from bot.features import compute_ofi, compute_atr
    from bot.scoring import compute_composite_scores
    from bot.execution import TBLManager

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    seed = 12345

    def run_pipeline() -> dict:
        np.random.seed(seed)
        df = _canonical_bar_snapshot(n=80, seed=seed)
        ofi = compute_ofi(df, window=20)
        atr = compute_atr(df, window=14)
        n = len(df)
        idx = df.index
        components = {
            "T_t": pd.Series(0.5, index=idx),
            "U_t": pd.Series(ofi.fillna(0.5).values, index=idx),
            "O_t": pd.Series(ofi.fillna(0.5).values, index=idx),
            "H_t": pd.Series(0.5, index=idx),
            "LDC_t": pd.Series(0.5, index=idx),
            "C_t": pd.Series(0.9, index=idx),
            "L_t": pd.Series(0.9, index=idx),
            "R_t": pd.Series(0.6, index=idx),
            "TBL_flag": pd.Series(0.5, index=idx),
            "OFI_rev": pd.Series(0.5, index=idx),
            "lambda_decay": pd.Series(0.5, index=idx),
        }
        composite, entry, exit_s, breakdown = compute_composite_scores(components)
        scores_arr = composite.dropna().values
        breakdown_arr = breakdown.dropna(how="all").values

        entry_price = float(df["close"].iloc[20])
        tp = 1.0
        sl = 1.0
        tmax = 10.0
        tbl = TBLManager(entry_price, tp, sl, tmax, entry_time=0.0)
        exits = []
        for i in range(21, min(35, len(df))):
            price = float(df["close"].iloc[i])
            ex, reason = tbl.on_tick(price, float(i))
            if ex:
                exits.append({"bar": i, "reason": reason})
                break

        return {
            "scores_hash": hashlib.sha256(scores_arr.tobytes()).hexdigest(),
            "breakdown_hash": hashlib.sha256(breakdown_arr.tobytes()).hexdigest(),
            "n_scores": len(scores_arr),
            "exits": exits,
        }

    result1 = run_pipeline()
    result2 = run_pipeline()
    assert result1["scores_hash"] == result2["scores_hash"]
    assert result1["breakdown_hash"] == result2["breakdown_hash"]
    assert result1["n_scores"] == result2["n_scores"]

    (ARTIFACTS_DIR / "integration_metrics.json").write_text(
        json.dumps({"runid": RUNID, "seed": seed, **result1}, indent=2)
    )


def test_integration_artifacts_dir_exists_after_run():
    """Integration run writes to validation/artifacts/<runid>/."""
    if not ARTIFACTS_DIR.exists():
        pytest.skip("run test_integration_features_scoring_execution_deterministic first")
    assert (ARTIFACTS_DIR / "integration_metrics.json").exists()
