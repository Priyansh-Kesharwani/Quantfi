import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, List

import yaml
from pydantic import BaseModel, Field


class BackendConfig(BaseModel):
    indicator_min_history_rows: int = 200
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_std_dev: float = 2.0
    atr_period: int = 14
    atr_percentile_lookback: int = 252
    z_score_periods: List[int] = Field(default_factory=lambda: [20, 50, 100])
    sma_short_period: int = 50
    sma_long_period: int = 200
    ema_period: int = 50
    adx_period: int = 14

    score_zone_strong_buy: float = 81.0
    score_zone_favorable: float = 61.0
    score_zone_neutral: float = 31.0
    score_default_value: float = 50.0
    score_top_factors_count: int = 3
    macro_fx_historical_avg: float = 83.0
    technical_rules: Dict[str, float] = Field(default_factory=lambda: {
        "sma_below_bonus": 15.0,
        "sma_above_penalty": 5.0,
        "rsi_oversold": 30.0,
        "rsi_low": 40.0,
        "rsi_overbought": 70.0,
        "rsi_high": 60.0,
        "rsi_oversold_bonus": 20.0,
        "rsi_low_bonus": 10.0,
        "rsi_overbought_penalty": 15.0,
        "rsi_high_penalty": 5.0,
        "macd_bull_bonus": 10.0,
        "macd_bear_penalty": 10.0,
        "bb_below_bonus": 15.0,
        "bb_near_multiplier": 1.02,
        "bb_near_bonus": 8.0,
        "adx_low": 20.0,
        "adx_high": 40.0,
        "adx_low_bonus": 5.0,
        "adx_high_penalty": 5.0,
    })
    volatility_rules: Dict[str, float] = Field(default_factory=lambda: {
        "atr_high_percentile": 80.0,
        "atr_mid_percentile": 60.0,
        "atr_low_percentile": 30.0,
        "atr_high_bonus": 20.0,
        "atr_mid_bonus": 10.0,
        "atr_low_penalty": 10.0,
        "drawdown_severe": -20.0,
        "drawdown_medium": -10.0,
        "drawdown_light": -5.0,
        "drawdown_flat": -1.0,
        "drawdown_severe_bonus": 30.0,
        "drawdown_medium_bonus": 20.0,
        "drawdown_light_bonus": 10.0,
        "drawdown_flat_penalty": 10.0,
    })
    statistical_rules: Dict[str, float] = Field(default_factory=lambda: {
        "extreme_low_z": -2.0,
        "strong_low_z": -1.5,
        "moderate_low_z": -1.0,
        "light_low_z": -0.5,
        "high_z": 1.5,
        "moderate_high_z": 1.0,
        "extreme_low_bonus": 50.0,
        "strong_low_bonus": 35.0,
        "moderate_low_bonus": 20.0,
        "light_low_bonus": 10.0,
        "high_penalty": 30.0,
        "moderate_high_penalty": 15.0,
    })
    macro_fx_rules: Dict[str, float] = Field(default_factory=lambda: {
        "high_dev": 5.0,
        "mid_dev": 2.0,
        "low_dev": -2.0,
        "very_low_dev": -5.0,
        "high_penalty": 20.0,
        "mid_penalty": 10.0,
        "low_bonus": 10.0,
        "very_low_bonus": 20.0,
    })
    phase1_score_bands: Dict[str, float] = Field(default_factory=lambda: {
        "favorable": 70.0,
        "slightly_favorable": 55.0,
        "neutral": 45.0,
        "slightly_unfavorable": 30.0,
    })
    phase1_trend_sma_window: int = 50

    symbol_aliases: Dict[str, str] = Field(default_factory=dict)
    exchange_suffixes: Dict[str, str] = Field(default_factory=dict)
    news_rss_base: str = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    news_rss_business: str = ""
    news_per_asset: int = 8

    fx_fallback_usd_inr: float = 83.5
    latest_price_period: str = "5d"
    history_period: str = "2y"
    news_lookback_days: int = 7
    news_page_size: int = 20
    news_process_limit: int = 50
    news_default_query: str = "gold OR silver OR federal reserve OR interest rates"

    cache_price_minutes: int = 5
    cache_indicators_hours: int = 1
    cache_scores_hours: int = 1
    cache_news_hours: int = 3
    backtest_padding_days: int = 300

    backtest_buy_dip_threshold_default: float = 60.0
    user_default_dca_amount: float = 5000.0
    user_default_dip_alert_threshold: float = 70.0

    llm_explain_provider: str = "openai"
    llm_explain_model: str = "gpt-5.2"
    llm_news_provider: str = "anthropic"
    llm_news_model: str = "claude-sonnet-4-5-20250929"
    llm_summary_max_chars: int = 100

    default_score_weights: Dict[str, float] = Field(default_factory=lambda: {
        "technical_momentum": 0.4,
        "volatility_opportunity": 0.2,
        "statistical_deviation": 0.2,
        "macro_fx": 0.2,
    })
    tuned_score_weights: Dict[str, float] = Field(default_factory=lambda: {
        "technical_momentum": 0.25,
        "volatility_opportunity": 0.25,
        "statistical_deviation": 0.35,
        "macro_fx": 0.15,
    })


def _load_yaml_backend_section() -> Dict[str, Any]:
    cfg_path = Path(__file__).parent.parent.parent / "config" / "settings.yml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r") as f:
        payload = yaml.safe_load(f) or {}
    result = payload.get("backend", {})
    dp = payload.get("data_providers", {})
    if dp.get("symbol_aliases"):
        result.setdefault("symbol_aliases", dp["symbol_aliases"])
    if dp.get("exchange_suffixes"):
        result.setdefault("exchange_suffixes", dp["exchange_suffixes"])
    if dp.get("news_rss_base"):
        result.setdefault("news_rss_base", dp["news_rss_base"])
    if dp.get("news_rss_business"):
        result.setdefault("news_rss_business", dp["news_rss_business"])
    if dp.get("news_per_asset"):
        result.setdefault("news_per_asset", dp["news_per_asset"])
    return result


def _env_override(cfg: BackendConfig) -> BackendConfig:
    overrides: Dict[str, Any] = {}
    if os.getenv("BACKEND_FX_FALLBACK_USD_INR"):
        overrides["fx_fallback_usd_inr"] = float(os.getenv("BACKEND_FX_FALLBACK_USD_INR"))
    if os.getenv("BACKEND_NEWS_PROCESS_LIMIT"):
        overrides["news_process_limit"] = int(os.getenv("BACKEND_NEWS_PROCESS_LIMIT"))
    if os.getenv("BACKEND_CACHE_PRICE_MINUTES"):
        overrides["cache_price_minutes"] = int(os.getenv("BACKEND_CACHE_PRICE_MINUTES"))
    if os.getenv("BACKEND_CACHE_INDICATORS_HOURS"):
        overrides["cache_indicators_hours"] = int(os.getenv("BACKEND_CACHE_INDICATORS_HOURS"))
    if os.getenv("BACKEND_CACHE_SCORES_HOURS"):
        overrides["cache_scores_hours"] = int(os.getenv("BACKEND_CACHE_SCORES_HOURS"))
    if os.getenv("BACKEND_CACHE_NEWS_HOURS"):
        overrides["cache_news_hours"] = int(os.getenv("BACKEND_CACHE_NEWS_HOURS"))
    if os.getenv("BACKEND_HISTORY_PERIOD"):
        overrides["history_period"] = os.getenv("BACKEND_HISTORY_PERIOD")
    if os.getenv("BACKEND_LATEST_PRICE_PERIOD"):
        overrides["latest_price_period"] = os.getenv("BACKEND_LATEST_PRICE_PERIOD")
    if os.getenv("BACKEND_PHASE1_TREND_SMA_WINDOW"):
        overrides["phase1_trend_sma_window"] = int(os.getenv("BACKEND_PHASE1_TREND_SMA_WINDOW"))
    if not overrides:
        return cfg
    return cfg.model_copy(update=overrides)


@lru_cache(maxsize=1)
def get_backend_config() -> BackendConfig:
    cfg = BackendConfig(**_load_yaml_backend_section())
    return _env_override(cfg)
