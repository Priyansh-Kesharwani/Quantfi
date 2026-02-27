from typing import Dict, List, Tuple
import numpy as np
from backend.models import ScoreBreakdown
from backend.app_config import get_backend_config
from scoring.composite import (
    compute_composite_score_single,
    technical_momentum_score as _tech_score,
    volatility_opportunity_score as _vol_score,
    statistical_deviation_score as _stat_score,
    macro_fx_score as _macro_score,
    SCORING_RULES,
)

CFG = get_backend_config()


class ScoringEngine:
    """Thin wrapper around the shared scoring module.

    The actual scoring rules live in ``scoring.composite`` so that
    the backtester's vectorized scorer and this engine stay in sync.
    This class adds human-readable factor explanations for the API.
    """

    DEFAULT_WEIGHTS = CFG.default_score_weights

    TUNED_WEIGHTS = CFG.tuned_score_weights

    @staticmethod
    def calculate_technical_momentum_score(indicators: Dict, current_price: float) -> Tuple[float, List[str]]:
        factors: List[str] = []
        sma_200 = indicators.get('sma_200')
        rsi = indicators.get('rsi_14')
        macd = indicators.get('macd')
        macd_hist = indicators.get('macd_hist')
        bb_lower = indicators.get('bb_lower')
        bb_upper = indicators.get('bb_upper')
        adx = indicators.get('adx_14')

        score = _tech_score(current_price, sma_200, rsi, macd, macd_hist, bb_lower, bb_upper, adx)

        if sma_200 and current_price < sma_200:
            pct_below = ((sma_200 - current_price) / sma_200) * 100
            factors.append(f"Price {pct_below:.1f}% below 200-day SMA (buying opportunity)")
        if rsi is not None:
            if rsi < 30:
                factors.append(f"RSI at {rsi:.1f} (oversold territory)")
            elif rsi < 40:
                factors.append(f"RSI at {rsi:.1f} (approaching oversold)")
        if macd is not None and macd_hist is not None:
            if macd < 0 and macd_hist > 0:
                factors.append("MACD showing bullish divergence")
        if bb_lower and current_price < bb_lower:
            factors.append("Price below lower Bollinger Band (extreme dip)")

        return score, factors

    @staticmethod
    def calculate_volatility_opportunity_score(indicators: Dict) -> Tuple[float, List[str]]:
        factors: List[str] = []
        atr_percentile = indicators.get('atr_percentile')
        drawdown = indicators.get('drawdown_pct')

        score = _vol_score(atr_percentile, drawdown)

        if atr_percentile and atr_percentile > 80:
            factors.append(f"Volatility at {atr_percentile:.0f}th percentile (high uncertainty = opportunity)")
        if drawdown is not None:
            if drawdown < -20:
                factors.append(f"Asset down {abs(drawdown):.1f}% from highs (significant dip)")
            elif drawdown < -10:
                factors.append(f"Asset down {abs(drawdown):.1f}% from highs (moderate dip)")

        return score, factors

    @staticmethod
    def calculate_statistical_deviation_score(indicators: Dict) -> Tuple[float, List[str]]:
        factors: List[str] = []
        z_scores = []
        if indicators.get('z_score_20'):
            z_scores.append(indicators['z_score_20'])
        if indicators.get('z_score_50'):
            z_scores.append(indicators['z_score_50'])
        if indicators.get('z_score_100'):
            z_scores.append(indicators['z_score_100'])

        avg_z = float(np.mean(z_scores)) if z_scores else None
        score = _stat_score(avg_z)

        if avg_z is not None:
            if avg_z < -2.0:
                factors.append(f"Price {abs(avg_z):.1f} std deviations below mean (extreme statistical dip)")
            elif avg_z < -1.5:
                factors.append(f"Price {abs(avg_z):.1f} std deviations below mean (strong statistical dip)")
            elif avg_z < -1.0:
                factors.append(f"Price {abs(avg_z):.1f} std deviations below mean (moderate statistical dip)")

        return score, factors

    @staticmethod
    def calculate_macro_fx_score(
        usd_inr_rate: float,
        historical_avg: float = CFG.macro_fx_historical_avg,
    ) -> Tuple[float, List[str]]:
        factors: List[str] = []
        score = _macro_score(usd_inr_rate)

        deviation_pct = ((usd_inr_rate - historical_avg) / historical_avg) * 100
        if deviation_pct > 5:
            factors.append(f"USD-INR at {usd_inr_rate:.2f}, {deviation_pct:.1f}% above average (expensive for INR buyers)")
        elif deviation_pct < -2:
            factors.append(f"USD-INR at {usd_inr_rate:.2f}, favorable for INR buyers")

        return score, factors

    @classmethod
    def calculate_composite_score(
        cls,
        indicators: Dict,
        current_price: float,
        usd_inr_rate: float,
        weights: Dict = None
    ) -> Tuple[float, ScoreBreakdown, List[str]]:
        if weights is None:
            weights = cls.DEFAULT_WEIGHTS

        tech_score, tech_factors = cls.calculate_technical_momentum_score(indicators, current_price)
        vol_score, vol_factors = cls.calculate_volatility_opportunity_score(indicators)
        stat_score, stat_factors = cls.calculate_statistical_deviation_score(indicators)
        macro_score, macro_factors = cls.calculate_macro_fx_score(usd_inr_rate)

        composite = (
            tech_score * weights['technical_momentum'] +
            vol_score * weights['volatility_opportunity'] +
            stat_score * weights['statistical_deviation'] +
            macro_score * weights['macro_fx']
        )

        breakdown = ScoreBreakdown(
            technical_momentum=tech_score,
            volatility_opportunity=vol_score,
            statistical_deviation=stat_score,
            macro_fx=macro_score
        )

        all_factors = tech_factors + vol_factors + stat_factors + macro_factors
        n = CFG.score_top_factors_count
        top_factors = all_factors[:n] if len(all_factors) >= n else all_factors

        return composite, breakdown, top_factors

    @staticmethod
    def get_zone(score: float) -> str:
        if score >= CFG.score_zone_strong_buy:
            return 'strong_buy'
        elif score >= CFG.score_zone_favorable:
            return 'favorable'
        elif score >= CFG.score_zone_neutral:
            return 'neutral'
        else:
            return 'unfavorable'
