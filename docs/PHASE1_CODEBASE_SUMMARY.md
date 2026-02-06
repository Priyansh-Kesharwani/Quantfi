# Phase 1 Codebase Summary

> Human-readable summary of the QuantFi codebase structure for Phase 1 implementation

## Project Overview

QuantFi is a **Financial Analysis Platform** for DCA (Dollar Cost Averaging) strategy optimization. The platform provides:

- Real-time price tracking for metals (Gold, Silver) and equities (US + Indian markets)
- Technical indicator calculation and visualization
- DCA favorability scoring to identify optimal accumulation periods
- News integration with LLM-powered classification
- Historical backtesting of DCA strategies

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend                             │
│              React + TailwindCSS (frontend/)                │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP/REST
┌─────────────────────────▼───────────────────────────────────┐
│                     Backend API                             │
│                FastAPI (backend/server.py)                  │
│  ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌───────────────┐ │
│  │ Assets   │ │ Indicators│ │ Scoring  │ │ Backtesting   │ │
│  └──────────┘ └───────────┘ └──────────┘ └───────────────┘ │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    Data Layer                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │   MongoDB    │  │   yfinance   │  │     NewsAPI      │   │
│  │   (motor)    │  │  (prices)    │  │   (articles)     │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## File Structure & Pointers

### Backend (`backend/`)

| File | Description | Key Classes/Functions |
|------|-------------|----------------------|
| **server.py** | FastAPI application, all REST endpoints | `app`, `api_router`, route handlers |
| **indicators.py** | Technical analysis indicators | `TechnicalIndicators` class |
| **scoring.py** | DCA composite scoring algorithm | `ScoringEngine` class |
| **models.py** | Pydantic data models | `Asset`, `PriceData`, `IndicatorData`, `DCAScore` |
| **data_providers.py** | External data fetching | `PriceProvider`, `FXProvider`, `NewsProvider` |
| **backtest.py** | DCA strategy backtesting | `BacktestEngine` class |
| **llm_service.py** | LLM integration for explanations | `LLMService` class |
| **requirements.txt** | Python dependencies | 140 pinned packages |

### Frontend (`frontend/`)

| Folder | Description |
|--------|-------------|
| **src/pages/** | Main pages (Dashboard, AssetDetail, BacktestLab, etc.) |
| **src/components/** | Reusable UI components |
| **src/api.js** | API client for backend communication |

### Tests (`tests/`)

| Item | Status |
|------|--------|
| `__init__.py` | Exists (empty placeholder) |
| Test files | **Not implemented** - Phase 1 will populate |
| CI/CD config | **Not found** - No `.github/workflows/` |

---

## Existing Indicators (backend/indicators.py)

The current `TechnicalIndicators` class provides:

1. **Trend Indicators**
   - SMA (Simple Moving Average) - 50, 200 day
   - EMA (Exponential Moving Average) - 50 day
   - ADX (Average Directional Index) - Trend strength

2. **Momentum Indicators**
   - RSI (Relative Strength Index) - 14 day
   - MACD (Moving Average Convergence Divergence)

3. **Volatility Indicators**
   - Bollinger Bands (20 day, 2 std dev)
   - ATR (Average True Range) - 14 day
   - ATR Percentile (rolling 252 day)

4. **Statistical Indicators**
   - Z-Score (20, 50, 100 day windows)
   - Drawdown Percentage

---

## Current Scoring System (backend/scoring.py)

The existing composite score uses **four components**:

| Component | Default Weight | What It Measures |
|-----------|---------------|------------------|
| Technical Momentum | 40% | Trend regime, RSI, MACD, BB position |
| Volatility Opportunity | 20% | ATR percentile, drawdown depth |
| Statistical Deviation | 20% | Z-score deviation from mean |
| Macro FX | 20% | USD-INR rate favorability |

**Score Range:** 0-100
**Zones:** unfavorable (<31), neutral (31-60), favorable (61-80), strong_buy (>80)

---

## Phase 1 Extension Points

### 1. New Indicators Package

**Location:** Create `indicators/` package at repo root (not inside backend/)

```
indicators/
├── __init__.py
├── hurst.py              # Wavelet-based Hurst exponent
├── hmm_regime.py         # HMM regime probability
├── vwap_z.py             # VWAP-based deviation
├── volatility.py         # Realized volatility
├── liquidity.py          # Amihud illiquidity ratio
├── coupling.py           # Systemic coupling (shrinkage covariance)
├── normalization.py      # ECDF → Z → Sigmoid pipeline
├── committee.py          # Robust aggregation (trimmed mean)
└── composite.py          # Phase 1 symbolic composite score
```

### 2. API Extension

**Add to `backend/server.py`:**
```python
# PHASE1: indicator hook - Add new router for Phase 1 dev endpoints
phase1_router = APIRouter(prefix="/dev/phase1")
# ... define Phase 1 endpoints ...
app.include_router(phase1_router)
```

### 3. Configuration

**Create:** `config/phase1.yml`
- Placeholder thresholds (symbolic, not numeric)
- Window size defaults
- `allow_production_mode: false` safety flag

### 4. Tests

**Create:** `tests/fixtures/` with deterministic CSV data
**Create:** Unit tests for each indicator module

---

## Key Design Decisions

### Why Separate `indicators/` Package?

1. **Modularity:** Phase 1 indicators are fundamentally different from basic TA
2. **Extensibility:** Easy to swap implementations (e.g., wavelet libraries)
3. **Testing:** Isolated testing with deterministic fixtures
4. **Documentation:** Clear separation for team understanding

### Why Symbolic Thresholds?

1. **No Premature Optimization:** Weights/thresholds require Phase 2 tuning
2. **Transparency:** Config file makes assumptions explicit
3. **Reproducibility:** All numeric choices documented

### Why Expanding ECDF?

1. **No Lookahead:** Percentiles computed only on past data
2. **Adaptive:** Automatically adjusts to new data
3. **Interpretable:** Scores always relative to historical distribution

---

## Dependencies Available for Phase 1

| Package | Version | Use Case |
|---------|---------|----------|
| `numpy` | 2.4.1 | Array operations |
| `pandas` | 2.3.3 | Time series |
| `scipy` | 1.17.0 | Statistical functions, optimization |
| `scikit-learn` | 1.5.0 | Ledoit-Wolf shrinkage, GMM |
| `PyYAML` | 6.0.3 | Config file parsing |

### Not Currently Installed (May Need)

| Package | Use Case | Fallback |
|---------|----------|----------|
| `pywt` (PyWavelets) | Wavelet Hurst | Deterministic R/S surrogate |

---

## Next Steps for Phase 1

1. ✅ Create `.cursor_memory/phase1_understanding.md`
2. ✅ Create `docs/PHASE1_CODEBASE_SUMMARY.md`
3. ⬜ Add extension point comments in existing files
4. ⬜ Create `indicators/` package with all modules
5. ⬜ Create `config/phase1.yml`
6. ⬜ Add `/dev/phase1/indicators/{asset}` endpoint
7. ⬜ Create test fixtures and unit tests
8. ⬜ Create documentation (IMPLEMENTATION.md, TESTING.md)
9. ⬜ Create TODOS_PHASE1.md
10. ⬜ Prepare git branch and PR

---

## Contact & References

- **Phase 1 Spec:** See user prompt in implementation task
- **Codebase Memory:** `.cursor_memory/phase1_understanding.md`
- **Main README:** `README.md` (project overview)

---

*Last Updated: 2026-02-07*
