from __future__ import annotations

from pathlib import Path
from typing import Any

from pathlib import Path

from backend.app_config import get_backend_config
from backend.indicators import TechnicalIndicators
from backend.scoring import ScoringEngine
from backend.backtest import BacktestEngine
from backend.llm_service import LLMService
from infrastructure.adapters.price_adapter import PriceAdapter
from infrastructure.adapters.fx_adapter import FXAdapter
from infrastructure.adapters.news_adapter import NewsAdapter
from services.asset_data_service import AssetDataService
from services.scoring_facade import ScoringFacade
from services.backtest_service import BacktestService
from services.news_service import NewsClassificationService
from services.simulation_service import SimulationService


class Container:
    def __init__(self, db: Any, project_root: Path) -> None:
        self._db = db
        self._project_root = Path(project_root)
        self._config = get_backend_config()
        self._price_adapter = PriceAdapter(self._config)
        self._fx_adapter = FXAdapter(self._config)
        self._news_adapter = NewsAdapter(self._config)
        self._llm_service = LLMService()
        self._indicators = TechnicalIndicators()
        self._scoring_engine = ScoringEngine()
        self._backtest_engine = BacktestEngine()

        self._asset_data_service = AssetDataService(
            self._price_adapter,
            self._fx_adapter,
            self._indicators,
            self._scoring_engine,
            self._llm_service,
            self._config,
        )
        self._scoring_facade = ScoringFacade(
            self._indicators,
            self._scoring_engine,
            self._price_adapter,
            self._fx_adapter,
            self._config,
        )
        self._backtest_service = BacktestService(
            self._price_adapter,
            self._backtest_engine,
            self._config,
        )
        self._news_service = NewsClassificationService(
            self._news_adapter,
            self._llm_service,
            self._config,
        )
        self._simulation_service = SimulationService(
            self._fx_adapter,
            self._config,
            self._project_root,
        )

    @property
    def db(self) -> Any:
        return self._db

    @property
    def config(self) -> Any:
        return self._config

    @property
    def price_adapter(self) -> Any:
        return self._price_adapter

    @property
    def fx_adapter(self) -> Any:
        return self._fx_adapter

    @property
    def news_adapter(self) -> Any:
        return self._news_adapter

    @property
    def asset_data_service(self) -> AssetDataService:
        return self._asset_data_service

    @property
    def scoring_facade(self) -> ScoringFacade:
        return self._scoring_facade

    @property
    def backtest_service(self) -> BacktestService:
        return self._backtest_service

    @property
    def news_service(self) -> NewsClassificationService:
        return self._news_service

    @property
    def simulation_service(self) -> SimulationService:
        return self._simulation_service

    @property
    def indicators(self) -> Any:
        return self._indicators

    @property
    def scoring_engine(self) -> Any:
        return self._scoring_engine

    @property
    def llm_service(self) -> Any:
        return self._llm_service
