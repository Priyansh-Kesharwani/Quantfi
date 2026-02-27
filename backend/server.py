import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient

from backend.container import Container
from backend.routes import create_api_router

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    mongo_url = os.environ["MONGO_URL"]
    db_name = os.environ["DB_NAME"]
    client = AsyncIOMotorClient(mongo_url)
    db = client[db_name]
    project_root = ROOT_DIR.parent
    app.state.container = Container(db, project_root)
    yield
    client.close()


def create_app(lifespan_=lifespan):
    app_ = FastAPI(title="Financial Analysis Platform", lifespan=lifespan_)
    app_.include_router(create_api_router())
    _cors_raw = os.environ.get("CORS_ORIGINS", "*")
    _cors_origins = (
        ["*"]
        if _cors_raw == "*"
        else [o.strip() for o in _cors_raw.split(",") if o.strip()]
    )
    app_.add_middleware(
        CORSMiddleware,
        allow_credentials=_cors_raw != "*",
        allow_origins=_cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app_


app = create_app()
