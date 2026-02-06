# Phase 1 TODO List

> Outstanding tasks and items requiring external configuration

## Critical TODOs

### Provider Keys Required

| Provider | Purpose | Where to Get | Config Location |
|----------|---------|--------------|-----------------|
| NewsAPI | News classification | https://newsapi.org | `config/phase1.yml` → `data_providers.api_keys.newsapi` |
| MCX | India commodity data | TBD (Phase 2) | `config/phase1.yml` → `data_providers.api_keys.mcx` |
| NSE | India equity data | TBD (Phase 2) | `config/phase1.yml` → `data_providers.api_keys.nse` |

### PyWavelets (pywt) Installation

The Hurst module uses R/S fallback. For better accuracy:

```bash
pip install PyWavelets
```

Then the wavelet-based Hurst estimator will be used automatically.

---

## Phase 2 Tasks (DO NOT IMPLEMENT NOW)

### 1. Numerical Weight Tuning

- [ ] Tune `regime_threshold` via sensitivity analysis
- [ ] Tune `S_scale` via score distribution analysis
- [ ] Calibrate `g_pers_params` from empirical Hurst distributions
- [ ] Run walk-forward backtests to validate parameters

### 2. HMM Enhancement

- [ ] Replace GMM stub with full Hidden Markov Model
- [ ] Implement t-distributed emissions for heavy tails
- [ ] Add jump penalty for regime transitions
- [ ] Implement Viterbi algorithm for state sequence

### 3. Data Provider Integration

- [ ] Implement MCX data provider for Indian commodities
- [ ] Implement NSE data provider for Indian equities
- [ ] Handle MCX/NSE idiosyncrasies (trading hours, holidays)
- [ ] Add real-time volume data feeds

### 4. Backtest Integration

- [ ] Integrate Phase 1 composite with existing backtest engine
- [ ] Implement slippage/cost modeling for DCA decisions
- [ ] Add transaction cost estimation
- [ ] Build walk-forward optimization framework

### 5. Production Readiness

- [ ] Enable `allow_production_mode` after validation
- [ ] Add monitoring/alerting for indicator calculations
- [ ] Implement caching layer for expensive computations
- [ ] Add rate limiting for API endpoints

---

## Code TODOs

### indicators/hurst.py
```
Line 25: TODO Phase 2: Implement wavelet-based estimator as primary method
```

### indicators/hmm_regime.py
```
Line 45: TODO Phase 2: Replace GMM with full HMM + t-emissions
Line 78: TODO Phase 2: Add jump penalty for regime switching
```

### indicators/vwap_z.py
```
Line 112: TODO: Implement proper VWAP data provider for production
```

### indicators/liquidity.py
```
Line 85: TODO: Connect to volume data provider
```

### indicators/coupling.py
```
Line 156: TODO: Connect to market data provider for benchmark returns
```

### config/phase1.yml
```
Line 15: TODO Phase 2: Tune regime_threshold via backtest
Line 28: TODO Phase 2: Tune S_scale via score distribution
Line 42: TODO Phase 2: Calibrate g_pers_params
```

### backend/phase1_routes.py
```
Line 89: TODO: Add authentication for production endpoints
```

---

## Safety Checklist Before Production

- [ ] All Phase 2 weight tuning complete
- [ ] Walk-forward backtest validation passed
- [ ] Provider API keys configured and tested
- [ ] `allow_production_mode: true` set in config
- [ ] Monitoring and alerting enabled
- [ ] Rate limiting configured
- [ ] Error handling reviewed
- [ ] Audit logging verified

---

## Questions for Phase 2 Planning

1. **Target Assets:** Which specific MCX/NSE instruments should be prioritized?
2. **Backtest Period:** What date range for walk-forward optimization?
3. **Rebalance Frequency:** How often should composite weights be recalibrated?
4. **Threshold Sensitivity:** What range of `regime_threshold` values to explore?
5. **Production Monitoring:** What metrics should trigger alerts?

---

*Created: 2026-02-07 | Updated: 2026-02-07*
