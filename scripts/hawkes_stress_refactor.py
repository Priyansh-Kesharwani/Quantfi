#!/usr/bin/env python3
"""
Refactor path — Hawkes stress: synthetic regimes, validate λ estimator RMSE ≤ config.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml


def main() -> int:
    from simulations.hawkes_simulator import run_all_regimes, simulate_regime, validate_estimation
    from indicators.hawkes import estimate_hawkes

    config_path = PROJECT_ROOT / "config" / "phase_refactor.yml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    rmse_tol = config.get("testing", {}).get("hawkes_rmse_tol", 0.1)

    regimes = [
        {"name": "bursty", "mu": 0.3, "alpha": 0.6, "beta": 1.0, "T": 50.0},
        {"name": "mild", "mu": 0.5, "alpha": 0.3, "beta": 1.0, "T": 50.0},
    ]
    results = run_all_regimes(regimes, base_seed=42, grid_resolution=0.5)
    for res in results:
        events = res["events"]
        timestamps = res["grid"]
        if len(events) < 5:
            continue
        intensity, _ = estimate_hawkes({"trades": events}, timestamps, decay=res.get("beta", 1.0))
        val = validate_estimation(intensity.values, res, rmse_threshold=rmse_tol)
        if not val.get("passed", True):
            print(f"Hawkes stress FAILED: {res['regime_name']} relative_rmse {val.get('relative_rmse', 0):.4f} > {rmse_tol}", file=sys.stderr)
            return 1
    print("Hawkes stress passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
