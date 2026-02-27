# API Overview

The backend exposes a REST API under the `/api` prefix. When the backend is running, interactive OpenAPI docs are available at:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Main endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/health | Health check |
| POST | /api/assets | Add asset to watchlist |
| GET | /api/assets | List active assets |
| DELETE | /api/assets/{symbol} | Soft-delete asset |
| GET | /api/prices/{symbol} | Latest price |
| GET | /api/prices/{symbol}/history | OHLCV history |
| GET | /api/indicators/{symbol} | Cached or fresh indicators |
| GET | /api/scores/{symbol} | DCA score and explanation |
| POST | /api/backtest | Run DCA backtest |
| GET | /api/news | Recent news |
| GET/PUT | /api/settings | User settings |
| GET | /api/dashboard | Aggregated assets, scores, prices, indicators |
| GET/POST | /api/sentiment/{symbol} | Sentiment analysis |
| GET | /api/simulation/templates | Strategy templates |
| GET | /api/simulation/cost-presets | Cost presets |
| POST | /api/simulation/run | Run portfolio simulation |

Request/response shapes and error codes are documented in the OpenAPI schema at `/docs`.
