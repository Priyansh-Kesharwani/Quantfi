# Deployment

## Environment Variables

- **MONGO_URL**: MongoDB connection string (e.g. `mongodb://localhost:27017` or Atlas URI).
- **DB_NAME**: Database name (e.g. `quantfi`).
- **CORS_ORIGINS**: Allowed origins for CORS (e.g. `http://localhost:3000` or `*`).
- **NEWS_API_KEY**: Optional; news falls back to RSS if unset or invalid.
- **EMERGENT_LLM_KEY** / **OPENAI_API_KEY**: Optional; for score explanation and news classification.

Optional overrides (prefix `BACKEND_`): `BACKEND_FX_FALLBACK_USD_INR`, `BACKEND_CACHE_PRICE_MINUTES`, `BACKEND_CACHE_INDICATORS_HOURS`, `BACKEND_CACHE_SCORES_HOURS`, `BACKEND_CACHE_NEWS_HOURS`, `BACKEND_HISTORY_PERIOD`, `BACKEND_LATEST_PRICE_PERIOD`, `BACKEND_NEWS_PROCESS_LIMIT`, `BACKEND_PHASE1_TREND_SMA_WINDOW`. See `backend/app_config.py`.

## Build and Run

1. **Backend**: From repository root, with Python 3.11+ and dependencies from `backend/requirements.txt` and optionally `requirements-bot.txt`:
   ```bash
   export PYTHONPATH="$PWD"
   uvicorn backend.server:app --host 0.0.0.0 --port 8000
   ```
   Or use `./start.sh`, which sets `PYTHONPATH` and runs the backend from the repo root.

2. **Frontend**: From `frontend/`:
   ```bash
   yarn install
   REACT_APP_BACKEND_URL=http://localhost:8000 yarn start
   ```
   Production build: `yarn build`; serve the `build/` directory with a static server.

3. **MongoDB**: Must be reachable at `MONGO_URL` before the backend starts.

## Production Considerations

- Set **CORS_ORIGINS** to the exact frontend origin(s); avoid `*` if credentials are used.
- Use a process manager (e.g. systemd, supervisord) or container orchestration for the backend and frontend.
- Keep secrets in environment or a secret manager; do not commit `backend/.env`.
- For high availability, use a MongoDB replica set and ensure the backend reconnects on failure (Motor handles reconnection within limits).
