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

```bash
python -m pytest tests -q                          # 96 backend/indicator unit tests
cd frontend && CI=true npx craco test --watchAll=false  # 38 component tests
npx playwright test                                 # E2E (requires app running)
```

Or all at once (unit + component):

```bash
npm run test:unit && npm run test:frontend
```

E2E tests (`e2e/app.spec.js`) run against the live app — start it first with `./start.sh`, then:

```bash
E2E_BASE_URL=http://localhost:3001 E2E_BACKEND_URL=http://localhost:8000 npx playwright test
```

## Project Structure

```
indicators/        Quantitative indicator library
backend/           FastAPI + MongoDB REST API
frontend/          React dashboard
backtester/        Walk-forward validation, signal sweep, portfolio sim
score_engine/      Composite scoring runner
data/              Fetcher + cache
config/            phase1.yml (all tunables)
tests/             pytest suite (96 tests)
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
