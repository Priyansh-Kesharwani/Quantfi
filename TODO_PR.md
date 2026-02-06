# Phase 1 Branch and PR Instructions

> Since direct GitHub integration is not available, follow these manual steps

## Branch Creation

```bash
# Navigate to project root
cd /path/to/Quantfi-main

# Create and checkout the Phase 1 branch
git checkout -b phase1/market-state-indicators

# Stage all Phase 1 files
git add .

# Initial commit
git commit -m "phase1: Add market state indicator engine scaffolding

- Add indicators/ package with Hurst, HMM, VWAP-Z, volatility, liquidity, coupling modules
- Add normalization pipeline (ECDF → Z → Sigmoid)
- Add committee aggregation for robust score combining
- Add composite score calculation (symbolic, no hardcoded weights)
- Add config/phase1.yml with placeholder configuration
- Add backend/phase1_routes.py for dev endpoints
- Add test fixtures in tests/fixtures/
- Add unit tests for all indicator modules
- Add documentation (PHASE1_IMPLEMENTATION.md, PHASE1_TESTING.md)
- Add codebase analysis in .cursor_memory/phase1_understanding.md
"

# Push to remote
git push -u origin phase1/market-state-indicators
```

## PR Title

```
[Phase 1] Market State Indicator Engine - Scaffolding & Symbolic Composite
```

## PR Body

```markdown
## Summary

This PR implements Phase 1 of the Composite DCA State Score system. The score describes market state (does NOT predict returns) to modulate DCA intensity. **Baseline DCA always stays ON** - the score only modifies intensity.

## What's Included

### New `indicators/` Package
- `hurst.py` - Wavelet/R-S Hurst exponent estimator (persistence measurement)
- `hmm_regime.py` - HMM regime detection (GMM stub for Phase 1)
- `vwap_z.py` - VWAP-based valuation Z-score
- `volatility.py` - Realized volatility and percentile
- `liquidity.py` - Amihud illiquidity measure
- `coupling.py` - Systemic coupling via Ledoit-Wolf shrinkage
- `normalization.py` - ECDF → Z → Sigmoid pipeline (no lookahead)
- `committee.py` - Robust aggregation (trimmed mean)
- `composite.py` - Symbolic composite calculation

### Composite Formula
```
Gate_t = C_t × L_t × R_t_thresholded
Opp_t = committee([T_t, U_t × g_pers(H_t)])
RawFavor_t = Opp_t × Gate_t
Score = 100 × clip(0.5 + (RawFavor - 0.5) × S_scale, 0, 1)
```

### Configuration
- `config/phase1.yml` - All numeric knobs are **SYMBOLIC PLACEHOLDERS**
- `allow_production_mode: false` - Safety flag prevents live data usage

### API Endpoint
- `GET /dev/phase1/indicators/{asset}` - Returns indicators + composite with metadata

### Tests
- `test_hurst.py` - Hurst estimation on synthetic fBm
- `test_vwap_z.py` - VWAP Z-score calculation
- `test_normalization.py` - ECDF pipeline (no lookahead verified)
- `test_composite_pipeline.py` - Full composite with anchor at 50

### Documentation
- `docs/PHASE1_IMPLEMENTATION.md` - Architecture and usage guide
- `docs/PHASE1_TESTING.md` - How to run tests
- `docs/PHASE1_CODEBASE_SUMMARY.md` - Codebase structure
- `.cursor_memory/phase1_understanding.md` - Persistent analysis
- `TODOS_PHASE1.md` - Outstanding items and Phase 2 tasks

## Non-Negotiable Rules (Verified)

✅ No prediction logic - score only modifies intensity
✅ No live data - uses fixtures only
✅ No hardcoded weights - all in config with null placeholders
✅ Reproducibility - seeded RNG in all tests
✅ Explainability - all functions return (value, meta) tuples
✅ Audit trail - logs to `logs/phase1_indicator_runs.json`

## How to Test

```bash
# From project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/ -v
```

## Phase 2 Tasks (NOT in this PR)

- Tune numeric weights via walk-forward backtest
- Replace GMM with full HMM + t-emissions
- Add wavelet Hurst (requires pywt)
- Connect real data providers (MCX/NSE)
- Enable production mode after validation

## Checklist

- [x] indicators/ package implemented
- [x] composite.py with symbolic formula
- [x] Backend endpoint for dev inspection
- [x] Unit tests with fixtures (no network)
- [x] Documentation updated
- [x] config/phase1.yml created
- [x] TODOS_PHASE1.md created
- [x] .cursor_memory/ analysis saved

## References

- Original prompt: See Phase 1 implementation task
- Codebase analysis: `.cursor_memory/phase1_understanding.md`
```

## Commit Convention

All commits on this branch should use the `phase1:` prefix:

```bash
git commit -m "phase1: Add Hurst estimator with R/S fallback"
git commit -m "phase1: Fix normalization edge case for short series"
git commit -m "phase1: Update documentation with API examples"
```

## After Merge

1. Delete branch:
```bash
git branch -d phase1/market-state-indicators
git push origin --delete phase1/market-state-indicators
```

2. Tag release:
```bash
git tag -a v0.1.0-phase1 -m "Phase 1: Market State Indicator Engine"
git push origin v0.1.0-phase1
```

---

*Generated: 2026-02-07*
