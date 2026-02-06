# Phase 1 Implementation Guide

> Market State Indicator Engine for DCA Modulation

## Overview

Phase 1 implements the **Composite_DCA_State_Score ∈ [0, 100]** that describes market state to modulate DCA (Dollar Cost Averaging) intensity. The key principle is that **baseline DCA always stays ON** - the score only modifies intensity, never stops DCA entirely.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           RAW DATA                                       │
│  (Price, Volume, Market Returns - from fixtures in Phase 1)             │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        INDICATOR MODULES                                 │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │
│  │  Hurst  │ │   HMM   │ │ VWAP Z  │ │   Vol   │ │Liquidity│           │
│  │  H_t    │ │   R_t   │ │  U_raw  │ │   V_t   │ │   L_t   │           │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘           │
│       │           │           │           │           │                 │
│       │           │           ▼           │           │    ┌─────────┐  │
│       │           │    ┌──────────────┐   │           │    │Coupling │  │
│       │           │    │Normalization │   │           │    │   C_t   │  │
│       │           │    │ECDF→Z→Sigmoid│   │           │    └────┬────┘  │
│       │           │    └──────┬───────┘   │           │         │       │
│       │           │           │           │           │         │       │
└───────┼───────────┼───────────┼───────────┼───────────┼─────────┼───────┘
        │           │           │           │           │         │
        ▼           ▼           ▼           ▼           ▼         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         COMPOSITE MODULE                                 │
│                                                                          │
│   g_pers(H_t) ────────────────┐                                         │
│                               ▼                                          │
│   T_t ────────────────┬─► Committee ─► Opp_t                            │
│   U_t × g_pers(H_t) ──┘     Aggregator                                  │
│                                          │                               │
│   C_t ────┬                             │                               │
│   L_t ────┼──► Gate_t ──────────────────┤                               │
│   R_t ────┘    (product)                │                               │
│                                          ▼                               │
│                            RawFavor_t = Opp_t × Gate_t                   │
│                                          │                               │
│                                          ▼                               │
│                    Score = 100 × clip(0.5 + (RF - 0.5) × S, 0, 1)       │
│                              (anchored at 50)                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
Quantfi-main/
├── indicators/                    # Phase 1 Indicator Package
│   ├── __init__.py               # Package exports
│   ├── hurst.py                  # Wavelet/R-S Hurst estimator
│   ├── hmm_regime.py             # HMM regime detection (GMM stub)
│   ├── vwap_z.py                 # VWAP-based valuation Z-score
│   ├── volatility.py             # Realized vol & percentile
│   ├── liquidity.py              # Amihud illiquidity
│   ├── coupling.py               # Systemic coupling (Ledoit-Wolf)
│   ├── normalization.py          # ECDF → Z → Sigmoid pipeline
│   ├── committee.py              # Robust aggregation
│   └── composite.py              # Final composite calculation
├── config/
│   └── phase1.yml                # Configuration (symbolic placeholders)
├── backend/
│   └── phase1_routes.py          # Dev API endpoints
├── tests/
│   ├── fixtures/                 # Deterministic CSV data
│   │   ├── synthetic_prices.csv
│   │   ├── trending_asset.csv
│   │   └── fbm_series.csv
│   ├── test_hurst.py
│   ├── test_vwap_z.py
│   ├── test_normalization.py
│   └── test_composite_pipeline.py
├── docs/
│   ├── PHASE1_IMPLEMENTATION.md  # This file
│   ├── PHASE1_TESTING.md         # Testing guide
│   └── PHASE1_CODEBASE_SUMMARY.md
├── logs/
│   └── phase1_indicator_runs.json  # Audit trail
└── .cursor_memory/
    └── phase1_understanding.md   # Persistent codebase analysis
```

## Indicator Modules

### 1. Hurst Exponent (`indicators/hurst.py`)

**Purpose:** Measure persistence/mean-reversion tendency

| H Value | Interpretation | DCA Implication |
|---------|----------------|-----------------|
| H > 0.5 | Persistent (trending) | Momentum strategies favored |
| H = 0.5 | Random walk | No predictable pattern |
| H < 0.5 | Mean-reverting | Contrarian strategies favored |

**Implementation:**
- Primary: Wavelet-based estimator (requires `pywt`)
- Fallback: Rescaled Range (R/S) method - always available, deterministic

**Usage:**
```python
from indicators.hurst import estimate_hurst

H_t, meta = estimate_hurst(prices, window=252, method="auto")
# H_t: time series of Hurst estimates
# meta: {"window_used", "n_obs", "method", "seed", "notes"}
```

### 2. HMM Regime Detection (`indicators/hmm_regime.py`)

**Purpose:** Compute P(state=StableExpansion) - probability of favorable regime

**Phase 1 Implementation:**
- GMM-based stub using sklearn.mixture.GaussianMixture
- Identifies "stable expansion" as state with positive mean + moderate variance

**Phase 2 TODO:**
- Replace with full HMM + forward-backward algorithm
- Use t-distributed emissions for heavy tails
- Add jump penalty for regime switching

**Usage:**
```python
from indicators.hmm_regime import infer_regime_prob, HMMRegimeConfig

config = HMMRegimeConfig(n_states=2, window=252, seed=42)
R_t, meta = infer_regime_prob(prices, config)
```

### 3. VWAP Z-Score (`indicators/vwap_z.py`)

**Purpose:** Measure deviation from fair value (VWAP)

| Z Value | Interpretation |
|---------|----------------|
| Z < -2 | Extremely undervalued |
| Z < -1 | Moderately undervalued |
| -1 < Z < 1 | Fair value |
| Z > 1 | Overvalued |

**Fallback:** SMA-based Z when volume data unavailable

**Usage:**
```python
from indicators.vwap_z import compute_vwap_z

Z_t, meta = compute_vwap_z(prices, volumes, window=20)
# Negative Z → undervalued → favorable for DCA
```

### 4. Volatility Regime (`indicators/volatility.py`)

**Purpose:** Assess volatility regime (calm vs stressed)

- Computes realized volatility (rolling std of returns, annualized)
- Computes expanding percentile rank
- Returns score where HIGH = calm market (favorable)

**Usage:**
```python
from indicators.volatility import volatility_regime_score

V_t, meta = volatility_regime_score(prices, vol_window=21, pct_lookback=252)
```

### 5. Liquidity (`indicators/liquidity.py`)

**Purpose:** Measure liquidity via Amihud illiquidity ratio

- High illiquidity = high price impact = unfavorable
- Score inverted: HIGH = liquid (favorable)

**Usage:**
```python
from indicators.liquidity import liquidity_score

L_t, meta = liquidity_score(prices, volumes, window=21)
```

### 6. Systemic Coupling (`indicators/coupling.py`)

**Purpose:** Measure correlation with market/other assets

**Why Shrinkage Covariance?**
- Sample covariance is noisy with limited data
- Ledoit-Wolf shrinkage provides regularized estimate
- More robust during regime changes

**Usage:**
```python
from indicators.coupling import coupling_score

C_t, meta = coupling_score(asset_returns, market_returns=None, window=63)
# Returns neutral (0.5) when market data unavailable
```

### 7. Normalization (`indicators/normalization.py`)

**Pipeline:** Raw → Expanding ECDF → Inverse Normal → Sigmoid → Polarity Align

**Key Property:** No lookahead - percentiles use only past data

**Usage:**
```python
from indicators.normalization import normalize_to_score

score, meta = normalize_to_score(raw_values, min_obs=100, k=1.0, higher_is_favorable=True)
```

### 8. Committee Aggregation (`indicators/committee.py`)

**Purpose:** Robust aggregation reducing outlier influence

**Methods:**
- `trimmed_mean` (default): Remove top/bottom 10%
- `winsorized_mean`: Cap extreme values
- `median`: Most robust
- `weighted`: Custom weights

**Usage:**
```python
from indicators.committee import agg_committee

agg, meta = agg_committee([0.7, 0.3, 0.8, 0.75], method="trimmed_mean")
```

### 9. Composite Score (`indicators/composite.py`)

**Formula:**
1. `Gate_t = C_t × L_t × R_t_thresholded`
2. `g_pers(H_t)` → persistence modifier
3. `Opp_t = committee([T_t, U_t × g_pers(H_t)])`
4. `RawFavor_t = Opp_t × Gate_t`
5. `Score = 100 × clip(0.5 + (RawFavor - 0.5) × S_scale, 0, 1)`

**Anchor:** Score = 50 when RawFavor = 0.5

**Usage:**
```python
from indicators.composite import compute_composite_score, Phase1Config

config = Phase1Config(S_scale=None, regime_threshold=None)  # Symbolic
result = compute_composite_score(
    T_t=0.6, U_t=0.7, V_t=0.5, L_t=0.8, C_t=0.7, H_t=0.55, R_t=0.6,
    config=config
)
print(f"Score: {result.score}, Gate: {result.Gate_t}, Opp: {result.Opp_t}")
```

## Configuration (`config/phase1.yml`)

All numeric knobs are **SYMBOLIC PLACEHOLDERS**:

```yaml
# SYMBOLIC - will be tuned in Phase 2
regime_threshold: null
S_scale: null
g_pers_params:
  H_neutral: null
  H_favorable: null
  H_unfavorable: null

# Safety flag
allow_production_mode: false
```

## API Endpoint

**GET /dev/phase1/indicators/{asset}**

Returns:
- All component values (raw + normalized)
- Composite calculation steps
- Metadata for audit

## Explainability Requirements

Every indicator function returns `(value, meta)` where `meta` includes:
- `window_used`
- `n_obs`
- `method`
- `seed` (if applicable)
- `notes`

Composite logs to `logs/phase1_indicator_runs.json`:
```json
{
  "timestamp": "2026-02-07T12:00:00",
  "composite_score": 45.2,
  "components": {"T_t": 0.6, "U_t": 0.7, ...},
  "intermediates": {"Gate_t": 0.3, "Opp_t": 0.5, ...},
  "config": {...}
}
```

## Phase 2 Enhancements

| Item | Current | Phase 2 |
|------|---------|---------|
| HMM | GMM stub | Full HMM + t-emissions |
| Hurst | R/S fallback | Wavelet primary (pywt) |
| Thresholds | Symbolic (null) | Tuned via backtest |
| Data | Fixtures | Live providers |
| Weights | Placeholder | Walk-forward optimization |

## Non-Negotiable Rules

1. **No Prediction:** Score only modifies intensity, never stops DCA
2. **No Live Data:** Use fixtures until `allow_production_mode: true`
3. **No Hardcoded Weights:** All numeric knobs in config
4. **Reproducibility:** Seeded RNG in tests
5. **Explainability:** All functions return metadata

---

*Created: 2026-02-07 | Phase 1 Implementation*
