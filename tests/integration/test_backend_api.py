import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MONGO_URL", "mongodb://localhost:27017")
os.environ.setdefault("DB_NAME", "quantfi_test")


@pytest.fixture
def mock_db():
    db = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.to_list = AsyncMock(return_value=[])
    db.assets.find.return_value = mock_cursor
    db.assets.find_one = AsyncMock(return_value=None)
    db.assets.insert_one = AsyncMock()
    db.assets.update_one = AsyncMock(return_value=MagicMock(modified_count=1))
    db.user_settings.find_one = AsyncMock(return_value=None)
    db.user_settings.insert_one = AsyncMock()
    db.user_settings.update_one = AsyncMock()
    db.price_history.find_one = AsyncMock(return_value=None)
    db.indicators.find_one = AsyncMock(return_value=None)
    db.scores.find_one = AsyncMock(return_value=None)
    mock_news_cursor = MagicMock()
    mock_news_cursor.sort.return_value.limit.return_value.to_list = (
        AsyncMock(return_value=[])
    )
    db.news_events.find.return_value = mock_news_cursor
    db.sentiment.find_one = AsyncMock(return_value=None)
    db.backtest_results.insert_one = AsyncMock()
    db.simulation_results.insert_one = AsyncMock()
    return db


@pytest.fixture
def app(mock_db):
    from starlette.requests import Request
    from backend.core.container import Container
    from backend.server import create_app
    from backend.routes import get_container

    test_container = Container(mock_db, PROJECT_ROOT)

    def override_get_container(req: Request) -> Container:
        return test_container

    @asynccontextmanager
    async def test_lifespan(app):
        app.state.container = test_container
        yield

    app_ = create_app(lifespan_=test_lifespan)
    app_.dependency_overrides[get_container] = override_get_container
    return app_


def test_health_route(app):
    from fastapi.testclient import TestClient
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "healthy"
    assert "timestamp" in data


def test_get_assets_empty(app):
    from fastapi.testclient import TestClient
    client = TestClient(app)
    response = client.get("/api/assets")
    assert response.status_code == 200
    assert response.json() == []


def test_add_asset_validation(app):
    from fastapi.testclient import TestClient
    client = TestClient(app)
    response = client.post(
        "/api/assets",
        json={"symbol": "", "name": "Test", "asset_type": "equity"},
    )
    assert response.status_code == 400
