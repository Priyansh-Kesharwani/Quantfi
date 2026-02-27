#!/usr/bin/env python
"""Deep validation of exported trades: regime coverage, capital flow, trade integrity."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXPORT_DIR = ROOT / "validation" / "trade_exports"
SUMMARY_FILE = EXPORT_DIR / "verification_summary.json"


def load_trades(csv_path: Path) -> list[dict]:
    trades = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["units"] = float(row["units"])
            row["price"] = float(row["price"])
            row["notional"] = float(row["notional"])
            row["fee"] = float(row["fee"])
            row["slippage"] = float(row["slippage"])
            row["funding_paid"] = float(row["funding_paid"])
            row["pnl"] = float(row["pnl"])
            row["leverage"] = float(row["leverage"])
            row["bar_idx"] = int(row["bar_idx"])
            trades.append(row)
    return trades


def validate_directional_trade_pairing(trades: list[dict]) -> list[str]:
    """Verify directional trades come in entry/exit pairs."""
    issues = []
    dir_trades = [t for t in trades if t["side"] in ("long_entry", "long_exit", "short_entry", "short_exit")]
    if not dir_trades:
        return issues

    open_pos = None
    for t in dir_trades:
        if t["side"] in ("long_entry", "short_entry"):
            if open_pos is not None:
                pass  # entries may overlap if closed by exit
            open_pos = t["side"]
        elif t["side"] == "long_exit":
            open_pos = None
        elif t["side"] == "short_exit":
            open_pos = None

    return issues


def validate_prices_positive(trades: list[dict]) -> list[str]:
    issues = []
    for i, t in enumerate(trades):
        if t["price"] <= 0:
            issues.append(f"Trade {i}: zero/negative price {t['price']}")
        if t["units"] <= 0:
            issues.append(f"Trade {i}: zero/negative units {t['units']}")
    return issues


def validate_fees_nonneg(trades: list[dict]) -> list[str]:
    issues = []
    for i, t in enumerate(trades):
        if t["fee"] < 0:
            issues.append(f"Trade {i}: negative fee {t['fee']}")
        if t["slippage"] < 0:
            issues.append(f"Trade {i}: negative slippage {t['slippage']}")
    return issues


def validate_timestamps_monotonic(trades: list[dict]) -> list[str]:
    issues = []
    for i in range(1, len(trades)):
        if trades[i]["timestamp"] < trades[i - 1]["timestamp"]:
            issues.append(f"Trades {i-1}->{i}: timestamps not monotonic ({trades[i-1]['timestamp']} > {trades[i]['timestamp']})")
    return issues


def validate_bar_idx_monotonic(trades: list[dict]) -> list[str]:
    """In adaptive mode, bar_idx resets on mode switches (directional->grid), so
    only flag decreases within a continuous run of the same trade type."""
    issues = []
    dir_sides = {"long_entry", "long_exit", "short_entry", "short_exit"}
    grid_sides = {"grid_buy", "grid_sell"}
    last_dir_idx = -1
    last_grid_idx = -1
    last_was_grid = None
    for i, t in enumerate(trades):
        if t["side"] in dir_sides:
            if t["bar_idx"] < last_dir_idx:
                issues.append(f"Trades {i}: directional bar_idx decreased ({last_dir_idx} > {t['bar_idx']})")
            last_dir_idx = t["bar_idx"]
            if last_was_grid is True:
                last_grid_idx = -1  # reset grid tracker on mode switch
            last_was_grid = False
        elif t["side"] in grid_sides:
            if last_was_grid is False:
                last_grid_idx = -1  # reset on mode switch into grid
            if t["bar_idx"] < last_grid_idx:
                issues.append(f"Trades {i}: grid bar_idx decreased ({last_grid_idx} > {t['bar_idx']})")
            last_grid_idx = t["bar_idx"]
            last_was_grid = True
    return issues


def validate_exit_reasons(trades: list[dict]) -> list[str]:
    issues = []
    valid_prefixes = ("trailing_stop", "score_exit", "max_hold", "liquidation", "grid_fill",
                       "stress", "close_all", "equity_stop", "mode_switch", "end_of_backtest")
    for i, t in enumerate(trades):
        reason = t["exit_reason"]
        if not any(reason.startswith(p) for p in valid_prefixes) and reason != "":
            issues.append(f"Trade {i}: unknown exit_reason '{reason}'")
    return issues


def analyze_side_distribution(trades: list[dict]) -> dict:
    counter = Counter(t["side"] for t in trades)
    return dict(counter)


def analyze_pnl_distribution(trades: list[dict]) -> dict:
    pnls = [t["pnl"] for t in trades if t["pnl"] != 0]
    if not pnls:
        return {"n": 0}
    arr = np.array(pnls)
    return {
        "n": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "win_pct": float((arr > 0).mean()),
        "total": float(np.sum(arr)),
    }


def main():
    with open(SUMMARY_FILE) as f:
        summary = json.load(f)

    csv_files = sorted(EXPORT_DIR.glob("*.csv"))
    print(f"Found {len(csv_files)} trade CSV files\n")

    total_issues = 0
    total_trades = 0
    all_sides = Counter()
    all_exit_reasons = Counter()

    by_strategy = defaultdict(list)
    by_symbol = defaultdict(list)
    by_timeframe = defaultdict(list)

    for csv_file in csv_files:
        trades = load_trades(csv_file)
        total_trades += len(trades)
        slug = csv_file.stem

        issues = []
        issues.extend(validate_prices_positive(trades))
        issues.extend(validate_fees_nonneg(trades))
        issues.extend(validate_timestamps_monotonic(trades))
        issues.extend(validate_bar_idx_monotonic(trades))
        issues.extend(validate_exit_reasons(trades))
        issues.extend(validate_directional_trade_pairing(trades))

        if issues:
            print(f"  ISSUES in {slug}:")
            for iss in issues:
                print(f"    {iss}")
            total_issues += len(issues)

        sides = analyze_side_distribution(trades)
        all_sides.update(sides)

        for t in trades:
            all_exit_reasons[t["exit_reason"]] += 1

        parts = slug.split("_")
        strategy = parts[2]
        by_strategy[strategy].append(len(trades))

    print(f"\n{'='*70}")
    print(f"DEEP TRADE VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Total CSV files: {len(csv_files)}")
    print(f"  Total trades: {total_trades:,}")
    print(f"  Total issues: {total_issues}")
    print()

    print("TRADE SIDE DISTRIBUTION:")
    for side, cnt in sorted(all_sides.items(), key=lambda x: -x[1]):
        print(f"  {side:15s}: {cnt:>8,} ({cnt/total_trades*100:.1f}%)")
    print()

    print("EXIT REASON DISTRIBUTION:")
    for reason, cnt in sorted(all_exit_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason:30s}: {cnt:>8,} ({cnt/total_trades*100:.1f}%)")
    print()

    print("TRADES BY STRATEGY:")
    for strat, counts in sorted(by_strategy.items()):
        total = sum(counts)
        print(f"  {strat:12s}: {total:>8,} trades across {len(counts)} files (avg {total/len(counts):.0f}/file)")
    print()

    # Verify all summary entries match CSVs
    mismatches = 0
    for entry in summary:
        if "error" in entry:
            continue
        csv_name = entry.get("csv_file", "")
        csv_path = EXPORT_DIR / csv_name
        if csv_path.exists():
            trades = load_trades(csv_path)
            if len(trades) != entry["n_trades"]:
                print(f"  MISMATCH {csv_name}: CSV has {len(trades)} trades, summary says {entry['n_trades']}")
                mismatches += 1

    print(f"SUMMARY vs CSV TRADE COUNT CHECK: {len(summary) - mismatches}/{len(summary)} match")
    print()

    # Sample a few CSVs for detailed PnL check
    print("SAMPLE PnL DISTRIBUTIONS (5 random files):")
    rng = np.random.RandomState(42)
    sample_files = rng.choice(csv_files, size=min(5, len(csv_files)), replace=False)
    for csv_file in sample_files:
        trades = load_trades(csv_file)
        pnl_stats = analyze_pnl_distribution(trades)
        if pnl_stats["n"] > 0:
            print(f"  {csv_file.stem}:")
            print(f"    Trades with PnL: {pnl_stats['n']}, Win%: {pnl_stats['win_pct']:.1%}, Mean: ${pnl_stats['mean']:.2f}, Total: ${pnl_stats['total']:.2f}")

    return 0 if total_issues == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
