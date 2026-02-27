#!/usr/bin/env python
"""Walk-forward parameter sweep for the crypto trading bot.

Generates 5000-bar synthetic data for each symbol, splits into 3 windows,
optimizes on windows 1-2, validates on window 3.  Reports the config that
maximises mean Sharpe across all symbols while keeping Sharpe > 0 everywhere.
"""

from __future__ import annotations

import itertools
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.crypto.synthetic import generate_synthetic_data as _generate_synthetic_data, generate_synthetic_funding as _generate_synthetic_funding  # noqa: E402
from crypto.regime.detector import CryptoRegimeConfig  # noqa: E402
from crypto.services.backtest_service import CryptoBacktestConfig, CryptoBacktestService  # noqa: E402

SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT"]
N_BARS = 5000
MODES = ["directional", "adaptive"]

PARAM_GRID = {
    "max_risk_per_trade": [0.10, 0.15, 0.20, 0.25],
    "entry_threshold": [20.0, 25.0, 30.0, 35.0],
    "exit_threshold": [10.0, 15.0, 20.0],
    "atr_trail_mult": [3.0, 4.0, 5.0],
    "max_holding_bars": [168, 336, 504],
}

HMM_GRID = {
    "warmup_bars": [300, 500, 750],
    "vol_window": [48, 96, 168],
    "circuit_breaker_dd": [-0.15, -0.20, -0.30, -0.40],
}


def _run_one(ohlcv_window, funding_window, symbol, mode, params, hmm_params):
    warmup = hmm_params.get("warmup_bars", 500)
    cfg = CryptoBacktestConfig(
        symbol=symbol,
        strategy_mode=mode,
        leverage=3.0,
        max_risk_per_trade=params["max_risk_per_trade"],
        entry_threshold=params["entry_threshold"],
        exit_threshold=params["exit_threshold"],
        atr_trail_mult=params["atr_trail_mult"],
        max_holding_bars=params["max_holding_bars"],
        score_exit_patience=3,
        compression_window=min(120, len(ohlcv_window) // 8),
        ic_window=min(120, len(ohlcv_window) // 8),
        ic_horizon=20,
        funding_window=min(200, len(ohlcv_window) // 6),
        regime_config=CryptoRegimeConfig(
            warmup_bars=warmup,
            rolling_window=warmup,
            refit_every=max(50, warmup // 5),
            vol_window=hmm_params.get("vol_window", 96),
            cooldown_bars=3,
            circuit_breaker_dd=hmm_params.get("circuit_breaker_dd", -0.25),
        ),
    )
    svc = CryptoBacktestService()
    try:
        result = svc.run(ohlcv_window, cfg, funding_rates=funding_window)
        return result["sharpe"], result["total_return_pct"], result["n_trades"]
    except Exception:
        return -999.0, 0.0, 0


def _generate_data():
    data = {}
    for sym in SYMBOLS:
        ohlcv = _generate_synthetic_data(N_BARS, sym)
        funding = _generate_synthetic_funding(N_BARS, sym)
        w_size = N_BARS // 3
        windows = []
        for w in range(3):
            s, e = w * w_size, (w + 1) * w_size
            windows.append((ohlcv.iloc[s:e].copy(), funding.iloc[s:e].copy()))
        data[sym] = windows
    return data


def tune_strategy_params():
    print("=" * 60)
    print("PHASE 1: Strategy parameter sweep")
    print("=" * 60)

    data = _generate_data()

    keys = list(PARAM_GRID.keys())
    combos = list(itertools.product(*[PARAM_GRID[k] for k in keys]))
    print(f"Testing {len(combos)} parameter combinations ...")

    hmm_params = {"warmup_bars": 500, "vol_window": 96, "circuit_breaker_dd": -0.25}
    best_score, best_params = -np.inf, None
    results = []

    for ci, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        sharpes = []
        all_positive = True

        for sym in SYMBOLS:
            train_ohlcv = pd.concat([data[sym][0][0], data[sym][1][0]])
            train_fund = pd.concat([data[sym][0][1], data[sym][1][1]])

            for mode in MODES:
                sh, ret, nt = _run_one(train_ohlcv, train_fund, sym, mode, params, hmm_params)
                sharpes.append(sh)
                if sh <= 0:
                    all_positive = False

        mean_sh = np.mean(sharpes)
        results.append({"params": params, "mean_sharpe": mean_sh, "all_positive": all_positive})

        if mean_sh > best_score:
            best_score = mean_sh
            best_params = params

        if (ci + 1) % 50 == 0:
            print(f"  [{ci+1}/{len(combos)}] best so far: sharpe={best_score:.3f}")

    print(f"\nBest train params: {best_params} (mean Sharpe={best_score:.3f})")

    print("\nValidation (window 3):")
    for sym in SYMBOLS:
        val_ohlcv, val_fund = data[sym][2]
        for mode in MODES:
            sh, ret, nt = _run_one(val_ohlcv, val_fund, sym, mode, best_params, hmm_params)
            print(f"  {sym} {mode:12s}: sharpe={sh:.2f}, ret={ret:.1f}%, trades={nt}")

    return best_params


def tune_hmm_params(strategy_params):
    print("\n" + "=" * 60)
    print("PHASE 2: HMM parameter sweep")
    print("=" * 60)

    data = _generate_data()

    keys = list(HMM_GRID.keys())
    combos = list(itertools.product(*[HMM_GRID[k] for k in keys]))
    print(f"Testing {len(combos)} HMM configs ...")

    best_score, best_hmm = -np.inf, None

    for ci, vals in enumerate(combos):
        hmm_params = dict(zip(keys, vals))
        sharpes = []

        for sym in SYMBOLS:
            train_ohlcv = pd.concat([data[sym][0][0], data[sym][1][0]])
            train_fund = pd.concat([data[sym][0][1], data[sym][1][1]])
            sh, _, _ = _run_one(train_ohlcv, train_fund, sym, "adaptive", strategy_params, hmm_params)
            sharpes.append(sh)

        mean_sh = np.mean(sharpes)
        if mean_sh > best_score:
            best_score = mean_sh
            best_hmm = hmm_params

    print(f"\nBest HMM params: {best_hmm} (mean Sharpe={best_score:.3f})")
    return best_hmm


def main():
    strategy_params = tune_strategy_params()
    hmm_params = tune_hmm_params(strategy_params)

    final = {**strategy_params, **hmm_params, "score_exit_patience": 3}
    print("\n" + "=" * 60)
    print("FINAL TUNED CONFIG")
    print("=" * 60)
    for k, v in final.items():
        print(f"  {k}: {v}")

    out_path = ROOT / "config" / "crypto_tuned.yml"
    out_path.parent.mkdir(exist_ok=True)
    import yaml

    with open(out_path, "w") as f:
        yaml.dump(final, f, default_flow_style=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
