#!/usr/bin/env bash
# Refactor path — full checklist: pytest, determinism, refactor runner, IC/decile, Hawkes stress, ablation.
# Exit non-zero if any step fails. Set REFACTOR_FULL=1 to run IC/decile, Hawkes stress, ablation.

set -e
cd "$(dirname "$0")/.."

echo "=== Refactor checks: pytest ==="
python3 -m pytest tests/test_normalization_refactor.py tests/test_component_formulas.py tests/test_composite_refactor.py tests/test_weights.py -v --tb=short 2>&1 || { echo "pytest failed"; exit 1; }

echo "=== Refactor checks: verify_refactor_determinism ==="
python3 scripts/verify_refactor_determinism.py --runid refactor_test 2>&1 || { echo "verify_refactor_determinism failed"; exit 1; }

echo "=== Refactor runner (artifacts) ==="
python3 -m validation.refactor_runner --runid refactor_test 2>&1 || { echo "refactor_runner failed"; exit 1; }

if [ -n "${REFACTOR_FULL:-}" ]; then
  echo "=== IC/decile (refactor) ==="
  python3 -m validation.run_ic_decile --out validation/reports/ic_refactor.json 2>&1 || { echo "run_ic_decile failed"; exit 1; }
  echo "=== Hawkes stress ==="
  python3 scripts/hawkes_stress_refactor.py 2>&1 || { echo "hawkes_stress_refactor failed"; exit 1; }
  echo "=== Ablation refactor ==="
  python3 scripts/ablation_refactor.py 2>&1 || { echo "ablation_refactor failed"; exit 1; }
fi

echo "=== All refactor checks passed ==="
