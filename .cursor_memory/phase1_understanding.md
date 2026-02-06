# Phase 1 Codebase Understanding

**Created:** 2026-02-07
**Purpose:** Persistent codebase analysis for Phase 1 Market State Indicator Engine

---

## Repository Overview

**Project:** QuantFi - Financial Analysis Platform for DCA (Dollar Cost Averaging) Strategy
**Primary Goal:** DCA Favorability Scoring to modulate investment intensity based on market state

---

## Backend Framework & Entrypoint

| Item | Location | Notes |
|------|----------|-------|
| **Framework** | FastAPI | Modern Python async web framework |
| **Entrypoint** | `backend/server.py` | Main app instance, all routes defined here |
| **Router Prefix** | `/api` | All endpoints under `/api/*` |
| **Database** | MongoDB (motor) | Async MongoDB client via motor |

### Key Backend Dependencies
- `fastapi==0.110.1` - Web framework
- `motor==3.3.1` - Async MongoDB driver
- `pandas==2.3.3` - Data manipulation
- `numpy==2.4.1` - Numerical computing
- `scikit-learn==1.5.0` - ML utilities (important for Phase 1 shrinkage estimators)
- `scipy==1.17.0` - Scientific computing
- `yfinance==0.2.48` - Market data fetching

---

## API Modules & Endpoints

### Current Endpoints (backend/server.py)
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/assets` | GET/POST | Asset watchlist management |
| `/api/assets/{symbol}` | DELETE | Remove asset |
| `/api/prices/{symbol}` | GET | Latest price |
| `/api/prices/{symbol}/history` | GET | Historical OHLCV |
| `/api/indicators/{symbol}` | GET | Technical indicators |
| `/api/scores/{symbol}` | GET | DCA favorability score |
| `/api/backtest` | POST | Run DCA backtest |
| `/api/news` | GET | Financial news |
| `/api/settings` | GET/PUT | User preferences |
| `/api/dashboard` | GET | Aggregate dashboard data |
| `/api/health` | GET | Health check |

### Extension Point for Phase 1
**Recommended location:** Add new router for `/dev/phase1/*` endpoints
**File:** `backend/server.py` (line ~546, after `app.include_router(api_router)`)

---

## Existing Indicator/Analytics Modules

### backend/indicators.py
**Class:** `TechnicalIndicators`
**Current Indicators:**
- `calculate_sma()` - Simple Moving Average
- `calculate_ema()` - Exponential Moving Average
- `calculate_rsi()` - Relative Strength Index
- `calculate_macd()` - MACD with signal and histogram
- `calculate_bollinger_bands()` - BB upper/middle/lower
- `calculate_atr()` - Average True Range
- `calculate_atr_percentile()` - Rolling ATR percentile
- `calculate_z_score()` - Standardized deviation from mean
- `calculate_drawdown()` - Drawdown from rolling high
- `calculate_adx()` - Average Directional Index
- `calculate_all_indicators()` - Aggregate function

**PHASE1 NOTE:** Existing indicators are basic TA. Phase 1 requires advanced indicators:
- Hurst exponent (persistence/mean-reversion)
- HMM regime detection
- VWAP-based valuation
- Liquidity (Amihud illiquidity)
- Systemic coupling (covariance shrinkage)

### backend/scoring.py
**Class:** `ScoringEngine`
**Current Composite Score:**
- `technical_momentum` (40% default weight)
- `volatility_opportunity` (20%)
- `statistical_deviation` (20%)
- `macro_fx` (20%)

**PHASE1 NOTE:** Phase 1 implements a fundamentally different composite:
- Normalized components via expanding ECDF
- Gate mechanism (Coupling × Liquidity × Regime threshold)
- Opportunity aggregator (Trend + Undervaluation × Persistence)
- Anchor at 50, configurable scale parameter

---

## Models (backend/models.py)

### Current Pydantic Models
- `Asset` - Watchlist item
- `PriceData` - OHLCV with currency conversion
- `IndicatorData` - Technical indicator values
- `ScoreBreakdown` - Component scores
- `DCAScore` - Composite score with explanation
- `NewsEvent` - Classified news
- `BacktestConfig` / `BacktestResult` - Backtest I/O
- `UserSettings` - Preferences

**PHASE1 NOTE:** Need new models for:
- Phase 1 indicator metadata (window_used, n_obs, method, etc.)
- Phase 1 composite score with intermediate values
- ECDF reference data

---

## Data Providers (backend/data_providers.py)

### Classes
- `PriceProvider` - yfinance wrapper with mock fallback
- `FXProvider` - USD-INR rate
- `NewsProvider` - NewsAPI wrapper

**PHASE1 NOTE:** Phase 1 must use fixture data when no provider keys present.
Mark TODOs for:
- MCX/NSE data providers
- Real-time volume data
- Multi-asset covariance data

---

## Tests Folder

**Location:** `tests/`
**Current State:** Empty (only `__init__.py` with placeholder)
**CI Config:** None found (no `.github/workflows`)

**PHASE1 REQUIREMENT:**
- Create `tests/fixtures/` for deterministic CSV data
- Create test modules: `test_hurst.py`, `test_vwap_z.py`, `test_normalization.py`, `test_composite_pipeline.py`
- All tests must use seeded RNG for reproducibility
- No network access in tests

---

## Package Manager

**File:** `backend/requirements.txt`
**Format:** Pinned versions (production-ready)

### Missing Dependencies for Phase 1
```
pywt  # PyWavelets for wavelet-based Hurst
# OR use deterministic surrogate if unavailable
```

**NOTE:** scikit-learn and scipy already present - sufficient for:
- Ledoit-Wolf shrinkage estimator (sklearn.covariance)
- GaussianMixture for HMM stub (sklearn.mixture)
- ECDF utilities

---

## Directory Structure

```
Quantfi-main/
├── backend/
│   ├── server.py          # FastAPI entrypoint (EXTENSION POINT)
│   ├── indicators.py      # Technical indicators (PHASE1: indicator hook)
│   ├── scoring.py         # Composite scoring (PHASE1: new composite)
│   ├── models.py          # Pydantic models
│   ├── data_providers.py  # Data fetching
│   ├── backtest.py        # Backtesting engine
│   ├── llm_service.py     # LLM integration
│   └── requirements.txt   # Python dependencies
├── frontend/              # React frontend
├── tests/                 # Test suite (empty, needs population)
├── memory/                # (empty)
├── test_reports/          # Test artifacts
└── README.md              # Project docs
```

---

## Extension Points Identified

### 1. backend/server.py
- **Line ~546:** Add Phase 1 router after main API router
- **Pattern:** Create `phase1_router = APIRouter(prefix="/dev/phase1")`

### 2. backend/indicators.py
- **Line ~176:** Add PHASE1 hook comment for new indicator integration
- **Pattern:** Import Phase 1 indicators module

### 3. backend/scoring.py
- **Line ~238:** Add PHASE1 hook for new composite algorithm
- **Pattern:** Import Phase 1 composite module

### 4. backend/models.py
- **After line 122:** Add Phase 1 specific models
- **Pattern:** New Pydantic models for Phase 1 data structures

---

## Phase 1 Implementation Plan

### New Package Structure
```
indicators/
├── __init__.py           # Package exports
├── hurst.py              # Wavelet-based Hurst estimator
├── hmm_regime.py         # HMM regime detection stub
├── vwap_z.py             # VWAP-based Z-score
├── volatility.py         # Realized volatility + percentile
├── liquidity.py          # Amihud illiquidity
├── coupling.py           # Systemic coupling (Ledoit-Wolf)
├── normalization.py      # ECDF → Z → Sigmoid pipeline
├── committee.py          # Robust aggregator (trimmed mean)
└── composite.py          # Symbolic composite score
```

### Config Structure
```
config/
└── phase1.yml            # Placeholder configuration
    - allow_production_mode: false
    - thresholds: null (symbolic)
    - S_scale: null (symbolic)
    - windows: configurable
```

### Test Structure
```
tests/
├── fixtures/
│   ├── synthetic_prices.csv
│   ├── synthetic_volume.csv
│   └── fbm_series.csv    # Fractional Brownian motion
├── test_hurst.py
├── test_vwap_z.py
├── test_normalization.py
└── test_composite_pipeline.py
```

---

## Critical Notes for Implementation

1. **No Prediction Logic:** Score only modifies DCA intensity, never stops baseline DCA
2. **No Live Data:** Use fixtures until explicitly enabled via config
3. **No Hardcoded Weights:** All numeric knobs in config with symbolic placeholders
4. **Explainability:** All indicators return (value, metadata) tuples
5. **Audit Trail:** Log all intermediate values to `logs/phase1_indicator_runs.json`
6. **Reproducibility:** Seeded RNG in all tests, deterministic fixtures

---

## Files Inspected

| File | Purpose | Lines |
|------|---------|-------|
| `backend/server.py` | FastAPI routes, entrypoint | 559 |
| `backend/indicators.py` | Technical indicators | 177 |
| `backend/scoring.py` | Composite scoring | 238 |
| `backend/models.py` | Pydantic models | 122 |
| `backend/data_providers.py` | Data fetching | 207 |
| `backend/backtest.py` | Backtesting engine | 118 |
| `backend/requirements.txt` | Dependencies | 140 |
| `tests/__init__.py` | Empty test package | 1 |
| `README.md` | Project README | (checked) |

---

**Last Updated:** 2026-02-07
**Author:** Phase 1 Implementation Assistant
