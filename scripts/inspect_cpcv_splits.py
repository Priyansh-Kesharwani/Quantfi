#!/usr/bin/env python3
"""
Print CPCV split boundaries (test_start_idx, test_end_idx, and dates if data loaded).
Use to verify test windows and purge/embargo without running full tuning.
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running from project root with PYTHONPATH or from scripts/
try:
    from validation.validator import CPCVConfig, generate_cpcv_splits
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from validation.validator import CPCVConfig, generate_cpcv_splits


def _ts_to_iso(ts) -> str:
    if hasattr(ts, "to_pydatetime"):
        return ts.to_pydatetime().date().isoformat()
    if hasattr(ts, "isoformat"):
        return str(ts)[:10]
    return str(ts)[:10]


def main():
    ap = argparse.ArgumentParser(description="Inspect CPCV split boundaries")
    ap.add_argument("--config", default="config/tuning_cpcv.yml", help="YAML config with data and cpcv")
    ap.add_argument("--with-dates", action="store_true", help="Load data to get date_index and print test_start_date, test_end_date")
    ap.add_argument("--out", default=None, help="Write split metadata JSON to this path")
    args = ap.parse_args()

    import yaml
    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    data_cfg = raw.get("data", {})
    cpcv_cfg = raw.get("cpcv", {})

    date_index = None
    n_samples = None
    if args.with_dates:
        try:
            from datetime import datetime
            from backtester.portfolio_simulator import prepare_multi_asset_data
            symbols = data_cfg.get("symbols", ["SPY"])
            if isinstance(symbols, str):
                symbols = [s.strip() for s in symbols.split(",")]
            start_s = data_cfg.get("start_date", "2018-01-01")
            end_s = data_cfg.get("end_date", "2024-01-01")
            start_dt = datetime.fromisoformat(start_s.replace("Z", "").split("T")[0])
            end_dt = datetime.fromisoformat(end_s.replace("Z", "").split("T")[0])
            date_index, _ = prepare_multi_asset_data(symbols, start_dt, end_dt, {})
            n_samples = len(date_index)
            print(f"Loaded data: {n_samples} bars, {date_index[0]} to {date_index[-1]}\n")
        except Exception as e:
            print(f"Could not load data for dates: {e}", file=sys.stderr)
            print("Proceeding with n_samples from config/estimate only.\n", file=sys.stderr)
    if n_samples is None:
        # Rough estimate: 252 * years
        start_s = data_cfg.get("start_date", "2018-01-01")
        end_s = data_cfg.get("end_date", "2024-01-01")
        y1, y2 = int(start_s[:4]), int(end_s[:4])
        n_samples = max(252 * (y2 - y1), 500)
        print(f"Using estimated n_samples={n_samples} (pass --with-dates to use real data)\n")

    cpcv_config = CPCVConfig.from_dict(cpcv_cfg)
    splits = generate_cpcv_splits(cpcv_config, n_samples)
    if not splits:
        print("No splits generated.")
        return 0

    records = []
    for s in splits:
        test_start_idx = int(s.test_idx[0]) if len(s.test_idx) else None
        test_end_idx = int(s.test_idx[-1]) if len(s.test_idx) else None
        rec = {"split_id": s.split_id, "test_start_idx": test_start_idx, "test_end_idx": test_end_idx}
        if date_index is not None and test_start_idx is not None and test_end_idx is not None:
            try:
                rec["test_start_date"] = _ts_to_iso(date_index[test_start_idx])
                rec["test_end_date"] = _ts_to_iso(date_index[test_end_idx])
            except Exception:
                pass
        records.append(rec)
        line = f"split_id={s.split_id} test_start_idx={test_start_idx} test_end_idx={test_end_idx}"
        if "test_start_date" in rec:
            line += f"  {rec['test_start_date']} → {rec['test_end_date']}"
        print(line)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
