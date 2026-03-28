from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient
from starlette.middleware.cors import CORSMiddleware

from backend.core.container import Container
from backend.routes import create_api_router

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_DEFAULT_MONGO_URL = "mongodb://localhost:27017"
_DEFAULT_DB_NAME = "quantfi"


@asynccontextmanager
async def lifespan(app: FastAPI):
    mongo_url = os.environ.get("MONGO_URL", _DEFAULT_MONGO_URL)
    db_name = os.environ.get("DB_NAME", _DEFAULT_DB_NAME)

    client = AsyncIOMotorClient(mongo_url)
    db = client[db_name]
    project_root = ROOT_DIR.parent

    app.state.container = Container(db, project_root)
    logger.info("Connected to MongoDB at %s (db=%s)", mongo_url, db_name)
    yield
    client.close()
    logger.info("MongoDB connection closed")


def create_app(lifespan_=lifespan) -> FastAPI:
    app_ = FastAPI(title="QuantFi", lifespan=lifespan_)
    app_.include_router(create_api_router())

    cors_raw = os.environ.get("CORS_ORIGINS", "*")
    cors_origins = (
        ["*"]
        if cors_raw == "*"
        else [o.strip() for o in cors_raw.split(",") if o.strip()]
    )
    app_.add_middleware(
        CORSMiddleware,
        allow_credentials=cors_raw != "*",
        allow_origins=cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app_


app = create_app()
