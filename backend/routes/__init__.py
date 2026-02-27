from fastapi import APIRouter, Request
from backend.container import Container


def get_container(req: Request) -> Container:
    return req.app.state.container


def create_api_router() -> APIRouter:
    from backend.routes import (
        assets,
        prices,
        indicators,
        scores,
        backtest,
        news,
        settings,
        dashboard,
        sentiment,
        simulation,
        health,
        crypto,
    )
    router = APIRouter(prefix="/api")
    router.include_router(assets.router, tags=["assets"])
    router.include_router(prices.router, tags=["prices"])
    router.include_router(indicators.router, tags=["indicators"])
    router.include_router(scores.router, tags=["scores"])
    router.include_router(backtest.router, tags=["backtest"])
    router.include_router(news.router, tags=["news"])
    router.include_router(settings.router, tags=["settings"])
    router.include_router(dashboard.router, tags=["dashboard"])
    router.include_router(sentiment.router, tags=["sentiment"])
    router.include_router(simulation.router, tags=["simulation"])
    router.include_router(health.router, tags=["health"])
    router.include_router(crypto.router, tags=["crypto"])
    return router
