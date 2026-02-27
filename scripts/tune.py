#!/usr/bin/env python3
"""
CLI for CPCV + GT-Score + MFBO tuning pipeline.

Usage:
    python scripts/tune.py --config config.yml [--output-dir DIR] [--seed N]
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BACKEND_DIR = _PROJECT_ROOT / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CPCV/MFBO tuning pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config (data, cpcv, mfbo, dsr_min)")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: from config or validation/outputs)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1

    import yaml
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}
    output_dir = args.output_dir or raw.get("output_dir") or str(_PROJECT_ROOT / "validation" / "outputs")
    output_path = Path(output_dir)

    from validation.orchestrator import run_orchestrator

    result = run_orchestrator(str(config_path), output_path, args.seed)
    print(f"Output dir: {result.artifact_paths.get('output_dir', output_path)}")
    if result.winning_config:
        print(f"Winning DSR: {result.winning_dsr}")
        print(f"Mean GT-Score: {result.winning_mean_gt}")
    else:
        print("No winning config (DSR < threshold or no trials)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
