from indicators.hurst import estimate_hurst, hurst_exponent

from indicators.hmm_regime import infer_regime_prob, regime_probability, HMMRegimeConfig

from indicators.vwap_z import compute_vwap_z

from indicators.volatility import realized_vol, volatility_percentile, volatility_regime_score

from indicators.liquidity import amihud_illiquidity, liquidity_score

from indicators.coupling import coupling_score, systemic_coupling

from indicators.normalization import (
    expanding_percentile,
    percentile_to_z,
    z_to_sigmoid,
    polarity_align,
    normalize_to_score,
    batch_normalize,
    expanding_ecdf_sigmoid,
)

from indicators.ofi import compute_ofi, compute_ofi_reversal

from indicators.hawkes import estimate_hawkes, hawkes_lambda_decay

from indicators.ldc import LDC, lorentzian_distance, build_templates_from_labels

from indicators.committee import agg_committee

from indicators.composite import (
    g_pers, compute_gate, compute_opportunity, compute_composite_score,
    Phase1Config, CompositeResult, Phase1Composite,
    compose_scores, PhaseAConfig, load_phaseA_config,
)

from indicators.trend import (
    trend_strength_score, adx_indicator, macd_histogram, ema_slope
)

from indicators.undervaluation import (
    undervaluation_score, price_vwap_zscore, drawdown_score
)

from indicators.indicator_engine import (
    IndicatorEngine, IndicatorConfig, IndicatorResult, compute_all_indicators
)

from indicators.geopolitics import (
    GeopoliticsEngine, GeopoliticsConfig, GeopoliticsResult,
    compute_geopolitical_score, get_G_t
)

from indicators.sentiment_agent import (
    SentimentAgent, SentimentResult, SentimentAgentConfig,
    AssetProfile, AssetProfileResolver,
    KnowledgeSource, KnowledgeCompiler,
    PromptValidator, SentimentBacktestValidator,
    compute_sentiment_G_t, full_sentiment_analysis,
    LLMConfig, CollectionLimits, TierWeights,
)

__all__ = [
    "estimate_hurst",
    "hurst_exponent",
    "infer_regime_prob",
    "regime_probability",
    "HMMRegimeConfig",
    "compute_vwap_z",
    "realized_vol",
    "volatility_percentile",
    "volatility_regime_score",
    "amihud_illiquidity",
    "liquidity_score",
    "systemic_coupling",
    "coupling_score",
    "expanding_percentile",
    "percentile_to_z",
    "z_to_sigmoid",
    "polarity_align",
    "normalize_to_score",
    "batch_normalize",
    "agg_committee",
    "g_pers",
    "compute_gate",
    "compute_opportunity",
    "compute_composite_score",
    "Phase1Config",
    "CompositeResult",
    "Phase1Composite",
    "trend_strength_score",
    "adx_indicator",
    "macd_histogram",
    "ema_slope",
    "undervaluation_score",
    "price_vwap_zscore",
    "drawdown_score",
    "IndicatorEngine",
    "IndicatorConfig",
    "IndicatorResult",
    "compute_all_indicators",
    "GeopoliticsEngine",
    "GeopoliticsConfig",
    "GeopoliticsResult",
    "compute_geopolitical_score",
    "get_G_t",
    "SentimentAgent",
    "SentimentResult",
    "SentimentAgentConfig",
    "AssetProfile",
    "AssetProfileResolver",
    "KnowledgeSource",
    "KnowledgeCompiler",
    "PromptValidator",
    "SentimentBacktestValidator",
    "compute_sentiment_G_t",
    "full_sentiment_analysis",
    "LLMConfig",
    "CollectionLimits",
    "TierWeights",
    # Phase A — Microstructure
    "expanding_ecdf_sigmoid",
    "compute_ofi",
    "compute_ofi_reversal",
    "estimate_hawkes",
    "hawkes_lambda_decay",
    "LDC",
    "lorentzian_distance",
    "build_templates_from_labels",
    "compose_scores",
    "PhaseAConfig",
    "load_phaseA_config",
]

__version__ = "1.1.0"  # Phase A
