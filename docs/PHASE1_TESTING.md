# Phase 1 Testing Guide

> How to run tests and validate the Phase 1 indicator engine

## Quick Start

```bash
# From project root
cd /path/to/Quantfi-main

# Run all Phase 1 tests
pytest tests/ -v

# Run specific test module
pytest tests/test_hurst.py -v
pytest tests/test_vwap_z.py -v
pytest tests/test_normalization.py -v
pytest tests/test_composite_pipeline.py -v

# Run with coverage
pytest tests/ --cov=indicators --cov-report=html
```

## Test Modules

### 1. `test_hurst.py`

**Purpose:** Validate Hurst exponent estimation

**Key Tests:**
- Random walk returns H ≈ 0.5
- Trending series returns H > 0.5
- Mean-reverting series returns H < 0.5
- Output shape matches input
- Warmup period returns NaN
- Deterministic with same seed

**Fixtures Used:**
- `tests/fixtures/trending_asset.csv`
- `tests/fixtures/fbm_series.csv`

### 2. `test_vwap_z.py`

**Purpose:** Validate VWAP Z-score calculation

**Key Tests:**
- Price below VWAP → negative Z
- Price above VWAP → positive Z
- Stable prices → Z near zero
- SMA fallback when no volume
- Undervaluation score inverts Z

**Fixtures Used:**
- `tests/fixtures/synthetic_prices.csv`

### 3. `test_normalization.py`

**Purpose:** Validate ECDF → Z → Sigmoid pipeline

**Key Tests:**
- No lookahead (percentiles use only past)
- Adding datapoint updates deterministically
- Warmup period is NaN
- Output in valid range (0, 1)
- Median percentile → Z = 0
- Zero Z → sigmoid = 0.5
- Polarity alignment works

### 4. `test_composite_pipeline.py`

**Purpose:** Validate full composite calculation

**Key Tests:**
- Output in [0, 100]
- Neutral inputs → score around 50
- Favorable inputs → score > baseline
- Unfavorable inputs → score < 50
- All components in result
- Deterministic with same inputs
- Config affects output

## Test Fixtures

### `tests/fixtures/synthetic_prices.csv`

50 rows of simple synthetic price data:
- Columns: `date`, `close`, `volume`, `high`, `low`, `open`
- Slight upward trend for testing

### `tests/fixtures/trending_asset.csv`

243 rows of strongly trending data:
- ~0.4% daily drift
- Suitable for testing high Hurst detection
- Full OHLCV data

### `tests/fixtures/fbm_series.csv`

120 rows of fractional Brownian motion-like data:
- True Hurst = 0.7 (persistent)
- Includes `true_hurst` column for validation

## Determinism Requirements

All tests MUST be deterministic:

```python
# Always set seed before generating random data
np.random.seed(42)

# Use the same seed across runs
def test_something():
    np.random.seed(42)
    data = np.random.randn(100)
    # ... test ...
```

## No Network Access

Tests MUST NOT access the network:

✅ **DO:**
```python
# Load from fixtures
fixture_path = PROJECT_ROOT / "tests" / "fixtures" / "data.csv"
df = pd.read_csv(fixture_path)
```

❌ **DON'T:**
```python
# Never fetch live data in tests
import yfinance as yf
data = yf.download("AAPL")  # FORBIDDEN
```

## Running Tests Locally

### Prerequisites

```bash
# Install test dependencies
pip install pytest pytest-cov

# Ensure indicators package is importable
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Full Test Suite

```bash
# All tests, verbose output
pytest tests/ -v

# Stop on first failure
pytest tests/ -v -x

# Show local variables on failure
pytest tests/ -v -l

# Run only tests matching pattern
pytest tests/ -v -k "hurst"
pytest tests/ -v -k "composite"
```

### With Coverage

```bash
# Generate coverage report
pytest tests/ --cov=indicators --cov-report=term-missing

# HTML report
pytest tests/ --cov=indicators --cov-report=html
# Open htmlcov/index.html in browser
```

### Specific Test Selection

```bash
# Single test file
pytest tests/test_hurst.py -v

# Single test class
pytest tests/test_hurst.py::TestRescaledRangeHurst -v

# Single test method
pytest tests/test_hurst.py::TestRescaledRangeHurst::test_random_walk_returns_around_0_5 -v
```

## Expected Test Output

```
tests/test_hurst.py::TestRescaledRangeHurst::test_random_walk_returns_around_0_5 PASSED
tests/test_hurst.py::TestRescaledRangeHurst::test_trending_series_returns_high_hurst PASSED
tests/test_hurst.py::TestRescaledRangeHurst::test_mean_reverting_series_returns_low_hurst PASSED
...
tests/test_composite_pipeline.py::TestCompositeScore::test_output_in_zero_to_hundred PASSED
tests/test_composite_pipeline.py::TestCompositeScore::test_neutral_inputs_return_around_fifty PASSED
...

==================== X passed in Y.YYs ====================
```

## Adding Real Provider Keys (Phase 2+)

When ready to test with real data:

1. Update `config/phase1.yml`:
```yaml
allow_production_mode: true  # Enable live data
data_providers:
  use_fixtures: false
  api_keys:
    newsapi: "your-api-key"
```

2. Set environment variables:
```bash
export NEWS_API_KEY="your-api-key"
export MCX_API_KEY="your-key"  # Phase 2
```

3. Create integration tests (separate from unit tests):
```python
# tests/integration/test_live_data.py
@pytest.mark.integration
@pytest.mark.skipif(not LIVE_DATA_ENABLED, reason="Live data disabled")
def test_with_live_data():
    # ...
```

## CI/CD Integration

### GitHub Actions (example)

```yaml
# .github/workflows/test.yml
name: Phase 1 Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r backend/requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: |
          export PYTHONPATH="${PYTHONPATH}:$(pwd)"
          pytest tests/ -v --cov=indicators
```

## Troubleshooting

### Import Errors

```bash
# Error: ModuleNotFoundError: No module named 'indicators'
# Fix: Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Fixture Not Found

```bash
# Error: FileNotFoundError: tests/fixtures/xyz.csv
# Fix: Run from project root
cd /path/to/Quantfi-main
pytest tests/ -v
```

### scipy/numpy Segfault

```bash
# Error: Exit code 139 (segfault)
# This can happen with certain numpy/scipy versions
# Fix: Update packages
pip install --upgrade numpy scipy scikit-learn
```

## Test Coverage Targets

| Module | Target Coverage |
|--------|-----------------|
| indicators/hurst.py | > 80% |
| indicators/vwap_z.py | > 80% |
| indicators/normalization.py | > 90% |
| indicators/composite.py | > 85% |
| indicators/committee.py | > 80% |

---

*Created: 2026-02-07 | Phase 1 Testing Guide*
