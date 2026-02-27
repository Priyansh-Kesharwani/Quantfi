# Refactor Summary

## Patterns Applied and Locations

| Pattern | Location | Purpose |
|--------|----------|---------|
| **Protocol (interface)** | `domain/protocols.py` | IPriceProvider, IFXProvider, INewsProvider, IIndicators, IScoringEngine, IBacktestEngine for dependency inversion |
| **Adapter** | `infrastructure/adapters/` (PriceAdapter, FXAdapter, NewsAdapter) | Wrap yfinance/NewsAPI behind domain protocols |
| **Facade** | `services/scoring_facade.py` (ScoringFacade) | Single entry for “indicators for symbol” and “score for symbol” |
| **Factory** | `backend/container.py` (Container), `weights/__init__.py` (WeightingFactory) | Build adapters and services in one place; create weighting instances from config |
| **Dependency Injection** | `backend/container.py`, `backend/routes/*` | Constructor injection for services; FastAPI Depends(get_container) for routes |
| **Config loader** | `infrastructure/config_loader.py` (ConfigLoader) | Centralized YAML loading for phase configs |

## Files Removed or Reorganized

- **backend/server.py**: Reduced to composition root only; all route handlers moved to `backend/routes/` (assets, prices, indicators, scores, backtest, news, settings, dashboard, sentiment, simulation, health).
- **backend/data_providers.py**: Now a thin delegation layer to `infrastructure.adapters`; singleton adapters created via `get_backend_config()`.
- **New packages**: `domain/`, `infrastructure/`, `services/` at repo root; `backend/routes/`, `backend/container.py` added.
- **Tests**: `tests/unit/`, `tests/integration/`, `tests/contract/` directories added (with `__init__.py`); `tests/integration/test_backend_api.py` added for API integration tests.

## Major Refactoring Steps

1. **Foundation**: Added `pyproject.toml` (Black, Ruff, mypy); domain Protocols; infrastructure adapters and config loader; services layer; test directory layout; frontend `jsconfig.json` with `checkJs`.
2. **Backend**: Introduced Container (DI), split server into route modules, wired routes via `create_api_router()`; backend internal imports switched to `backend.*`; `create_app(lifespan_=...)` for testability.
3. **Data providers**: Implemented PriceAdapter, FXAdapter, NewsAdapter in infrastructure; backend `data_providers` delegates to them for backward compatibility with BacktestEngine and existing code.
4. **Start script**: `start.sh` updated to run backend from repo root with `PYTHONPATH="$ROOT"` and `uvicorn backend.server:app`.
5. **Testing**: Backend API integration tests with mocked DB and dependency override for `get_container`.
6. **Documentation**: REFACTOR_SUMMARY, PATTERNS, DEPLOYMENT; README and ARCHITECTURE updated for new layout and run instructions.

## Baseline Architecture (Post-Refactor)

- **Interfaces**: FastAPI routers in `backend/routes/`; DTOs in `backend/models.py`.
- **Services**: AssetDataService, ScoringFacade, BacktestService, NewsClassificationService, SimulationService in `services/`.
- **Domain**: Protocols in `domain/protocols.py`; no framework or DB imports.
- **Infrastructure**: Adapters in `infrastructure/adapters/`; ConfigLoader in `infrastructure/config_loader.py`; backend `data_providers` and MongoDB used via Container.
