# QuantFi

DCA scoring platform. Scores assets on a 0–100 scale using technical, volatility, statistical, macro, sentiment, and regime indicators to modulate dollar-cost averaging intensity.

## Prerequisites

- Python 3.11+
- Node 18+ / Yarn
- MongoDB (local or Atlas)
- API keys: `NEWS_API_KEY`, `EMERGENT_LLM_KEY` (optional, degrades gracefully)

## Quick Start

```bash
# Configure backend/.env first
cat > backend/.env << EOF
MONGO_URL=mongodb://localhost:27017
DB_NAME=quantfi
NEWS_API_KEY=your_key_here
EMERGENT_LLM_KEY=your_key_here
CORS_ORIGINS=http://localhost:3000
EOF

# Start everything
./start.sh
```

Auto-detects port conflicts (e.g. if `:3000` is taken by Docker, picks the next free port). First run creates venv + installs deps automatically.

Override ports: `BACKEND_PORT=9000 PORT=4000 ./start.sh`

## Tests

Run from repository root (backend tests require `PYTHONPATH` for the refactored layout):

```bash
PYTHONPATH=. python3 -m pytest tests -q                    # Python unit and integration tests
cd frontend && CI=true npx craco test --watchAll=false      # Frontend component tests
npx playwright test                                         # E2E (requires app running)
```

Or via npm scripts (unit + frontend):

```bash
npm run test:unit && npm run test:frontend
```

E2E tests (`e2e/app.spec.js`) run against the live app — start it first with `./start.sh`, then:

```bash
E2E_BASE_URL=http://localhost:3001 E2E_BACKEND_URL=http://localhost:8000 npx playwright test
```

## Project Structure

Layered layout (see `docs/ARCHITECTURE.md` and `docs/PATTERNS.md`):

```
domain/            Protocols (IPriceProvider, IScoringEngine, etc.)
infrastructure/    Adapters (price, FX, news), config loader
services/          Use-case services and facades (scoring, backtest, news, simulation)
backend/           FastAPI app, routes, container (DI), models
backend/routes/    Route modules (assets, prices, indicators, scores, backtest, news, …)
indicators/        Quantitative indicator library
backtester/        Walk-forward validation, portfolio sim
frontend/          React dashboard
config/            YAML configs (phase1, phaseA, phaseB, phase3, tuning)
tests/             pytest (unit, integration, contract)
e2e/               Playwright E2E tests
```

## Config

Two sources, nothing else:

- `config/phase1.yml` — indicator windows, scoring rules, thresholds, weights
- `backend/.env` — secrets + runtime overrides

`backend/app_config.py` loads YAML → validates via Pydantic → applies env overrides (prefix `BACKEND_`). Cached at startup.

## API

Base: `http://localhost:8000/api`

```
POST   /assets                 Add to watchlist
GET    /assets                 List active
DELETE /assets/{symbol}        Soft-delete
GET    /prices/{symbol}        Latest (cached)
GET    /prices/{symbol}/history OHLCV
GET    /indicators/{symbol}    Technicals
GET    /scores/{symbol}        DCA score + explanation
POST   /backtest               Run backtest
GET    /news                   Classified events
GET    /settings               User prefs
PUT    /settings               Update prefs
GET    /dashboard              Aggregated view
GET    /health                 Status
```

Dev routes under `/dev/phase1/` for raw indicator inspection.

## Indicators

All in `indicators/`, same signature: `f(series, ...) → (ndarray, meta_dict)`.

```
hurst.py            Hurst exponent (R/S, wavelet)
hmm_regime.py       Bull/bear regime probability
vwap_z.py           VWAP Z-score
volatility.py       Realized vol percentile
liquidity.py        Amihud illiquidity
coupling.py         Market correlation
trend.py            ADX, MACD, EMA slope
undervaluation.py   Drawdown + VWAP inversion
geopolitics.py      GPR + GDELT
sentiment_agent.py  LLM sentiment (news, Reddit, blogs)
normalization.py    ECDF → Z → sigmoid → polarity
composite.py        Gate × Opportunity → score
```

## Supported Assets

Equities, ETFs, futures (`GC=F`, `SI=F`), crypto (`BTC-USD`), Indian markets (`RELIANCE.NS`, `GOLDBEES.NS`), currency pairs. Anything on Yahoo Finance.
