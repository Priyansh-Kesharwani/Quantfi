# Design Patterns

## Object Creation

- **Factory**: `WeightingFactory` in `weights/__init__.py` creates IC-EWMA or Kalman weighting from config. `Container` in `backend/container.py` acts as a composition root that builds all adapters and services.
- **Singleton (via DI)**: Single instances of adapters and services are created in `Container` and shared via `app.state.container` and `Depends(get_container)`; no module-level global services.

## API and External Systems

- **Adapter**: `PriceAdapter`, `FXAdapter`, `NewsAdapter` in `infrastructure/adapters/` implement domain protocols and wrap yfinance/NewsAPI so the rest of the app depends on interfaces, not concrete APIs.
- **Facade**: `ScoringFacade` in `services/scoring_facade.py` provides “get indicators for symbol” and “get score for symbol” so route handlers call one facade instead of orchestrating indicators, scoring, and price/FX providers directly.

## Service Logic and Dependency Flow

- **Dependency Injection**: All services and adapters receive dependencies via constructors. The composition root is `backend/container.py`; routes obtain the container via `get_container(req: Request)` and use its services.
- **Strategy (optional)**: Validation and tuning can use pluggable strategies (e.g. CPCV vs walkforward) behind a common interface; current code still uses existing validation modules.

## Configuration

- **Config loader**: `ConfigLoader` in `infrastructure/config_loader.py` centralizes YAML loading; backend continues to use `backend.app_config.get_backend_config()` which can be backed by the same loader or existing YAML + env logic.
