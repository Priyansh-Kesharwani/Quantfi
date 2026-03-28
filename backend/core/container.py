from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.core.config import get_backend_config
from backend.core.protocols import (
    IPriceProvider,
    IFXProvider,
    INewsProvider,
    IIndicators,
    IScoringEngine,
    IBacktestEngine,
    ILLMService,
)

from backend.adapters.price_adapter import PriceAdapter
from backend.adapters.fx_adapter import FXAdapter
from backend.adapters.news_adapter import NewsAdapter

from backend.repositories.asset_repository import AssetRepository
from backend.repositories.price_repository import PriceRepository
from backend.repositories.indicator_repository import IndicatorRepository
from backend.repositories.score_repository import ScoreRepository
from backend.repositories.news_repository import NewsRepository
from backend.repositories.settings_repository import SettingsRepository
from backend.repositories.paper_trading_repository import PaperTradingRepository

from backend.services.indicator_service import TechnicalIndicators
from backend.services.scoring_service import ScoringEngine, ScoringFacade
from backend.services.backtest_service import BacktestEngine, BacktestService
from backend.services.llm_service import LLMService
from backend.services.asset_service import AssetDataService
from backend.services.news_service import NewsClassificationService
from backend.services.simulation_service import SimulationService
from backend.services.dashboard_service import DashboardService
from backend.services.price_service import PriceService
from backend.services.sentiment_service import SentimentService
from backend.services.crypto_service import CryptoService
from backend.services.paper_trading_service import PaperTradingService


class Container:
    def __init__(self, db: Any, project_root: Path) -> None:
        self._db = db
        self._project_root = Path(project_root)
        self._config = get_backend_config()

        # --- adapters ---
        self._price_adapter: IPriceProvider = PriceAdapter(self._config)
        self._fx_adapter: IFXProvider = FXAdapter(self._config)
        self._news_adapter: INewsProvider = NewsAdapter(self._config)

        # --- core engine pieces ---
        self._llm_service: ILLMService = LLMService()
        self._indicators: IIndicators = TechnicalIndicators()
        self._scoring_engine: IScoringEngine = ScoringEngine()
        self._backtest_engine: IBacktestEngine = BacktestEngine(fx_provider=self._fx_adapter)

        # --- repositories ---
        self._asset_repo = AssetRepository(db)
        self._price_repo = PriceRepository(db)
        self._indicator_repo = IndicatorRepository(db)
        self._score_repo = ScoreRepository(db)
        self._news_repo = NewsRepository(db)
        self._settings_repo = SettingsRepository(db)
        self._paper_trading_repo = PaperTradingRepository(db)

        # --- services ---
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
        self._dashboard_service = DashboardService(
            self._asset_repo,
            self._score_repo,
            self._price_repo,
            self._indicator_repo,
        )
        self._price_service = PriceService(
            self._price_adapter,
            self._fx_adapter,
            self._price_repo,
            self._asset_repo,
            self._config,
        )
        self._sentiment_service = SentimentService(
            db=self._db,
            config=self._config,
        )
        self._crypto_service = CryptoService()
        self._paper_trading_service = PaperTradingService(
            self._paper_trading_repo,
            self._price_adapter,
        )

    # --- db & config ---
    @property
    def db(self) -> Any:
        return self._db

    @property
    def config(self) -> Any:
        return self._config

    # --- adapters ---
    @property
    def price_adapter(self) -> IPriceProvider:
        return self._price_adapter

    @property
    def fx_adapter(self) -> IFXProvider:
        return self._fx_adapter

    @property
    def news_adapter(self) -> INewsProvider:
        return self._news_adapter

    # --- repositories ---
    @property
    def asset_repo(self) -> AssetRepository:
        return self._asset_repo

    @property
    def price_repo(self) -> PriceRepository:
        return self._price_repo

    @property
    def indicator_repo(self) -> IndicatorRepository:
        return self._indicator_repo

    @property
    def score_repo(self) -> ScoreRepository:
        return self._score_repo

    @property
    def news_repo(self) -> NewsRepository:
        return self._news_repo

    @property
    def settings_repo(self) -> SettingsRepository:
        return self._settings_repo

    @property
    def paper_trading_repo(self) -> PaperTradingRepository:
        return self._paper_trading_repo

    # --- core engine pieces ---
    @property
    def indicators(self) -> IIndicators:
        return self._indicators

    @property
    def scoring_engine(self) -> IScoringEngine:
        return self._scoring_engine

    @property
    def llm_service(self) -> ILLMService:
        return self._llm_service

    # --- services ---
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
    def dashboard_service(self) -> DashboardService:
        return self._dashboard_service

    @property
    def price_service(self) -> PriceService:
        return self._price_service

    @property
    def sentiment_service(self) -> SentimentService:
        return self._sentiment_service

    @property
    def crypto_service(self) -> CryptoService:
        return self._crypto_service

    @property
    def paper_trading_service(self) -> PaperTradingService:
        return self._paper_trading_service
