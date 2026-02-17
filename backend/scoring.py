from typing import Dict, List, Tuple
import numpy as np
from models import ScoreBreakdown
from app_config import get_backend_config

CFG = get_backend_config()
TECH = CFG.technical_rules
VOL = CFG.volatility_rules
STAT = CFG.statistical_rules
FX = CFG.macro_fx_rules


class ScoringEngine:
    
    DEFAULT_WEIGHTS = CFG.default_score_weights
    
    TUNED_WEIGHTS = CFG.tuned_score_weights
    
    @staticmethod
    def calculate_technical_momentum_score(indicators: Dict, current_price: float) -> Tuple[float, List[str]]:
        factors = []
        score = CFG.score_default_value
        
        if indicators.get('sma_200'):
            if current_price < indicators['sma_200']:
                score += TECH['sma_below_bonus']
                pct_below = ((indicators['sma_200'] - current_price) / indicators['sma_200']) * 100
                factors.append(f"Price {pct_below:.1f}% below 200-day SMA (buying opportunity)")
            else:
                score -= TECH['sma_above_penalty']
        
        rsi = indicators.get('rsi_14')
        if rsi:
            if rsi < TECH['rsi_oversold']:
                score += TECH['rsi_oversold_bonus']
                factors.append(f"RSI at {rsi:.1f} (oversold territory)")
            elif rsi < TECH['rsi_low']:
                score += TECH['rsi_low_bonus']
                factors.append(f"RSI at {rsi:.1f} (approaching oversold)")
            elif rsi > TECH['rsi_overbought']:
                score -= TECH['rsi_overbought_penalty']
            elif rsi > TECH['rsi_high']:
                score -= TECH['rsi_high_penalty']
        
        macd_hist = indicators.get('macd_hist')
        macd = indicators.get('macd')
        if macd_hist and macd:
            if macd < 0 and macd_hist > 0:
                score += TECH['macd_bull_bonus']
                factors.append("MACD showing bullish divergence")
            elif macd > 0 and macd_hist < 0:
                score -= TECH['macd_bear_penalty']
        
        bb_lower = indicators.get('bb_lower')
        bb_upper = indicators.get('bb_upper')
        if bb_lower and bb_upper:
            if current_price < bb_lower:
                score += TECH['bb_below_bonus']
                factors.append("Price below lower Bollinger Band (extreme dip)")
            elif current_price < bb_lower * TECH['bb_near_multiplier']:
                score += TECH['bb_near_bonus']
        
        adx = indicators.get('adx_14')
        if adx:
            if adx < TECH['adx_low']:
                score += TECH['adx_low_bonus']
            elif adx > TECH['adx_high']:
                score -= TECH['adx_high_penalty']
        
        return max(0, min(100, score)), factors
    
    @staticmethod
    def calculate_volatility_opportunity_score(indicators: Dict) -> Tuple[float, List[str]]:
        factors = []
        score = CFG.score_default_value
        
        atr_percentile = indicators.get('atr_percentile')
        if atr_percentile:
            if atr_percentile > VOL['atr_high_percentile']:
                score += VOL['atr_high_bonus']
                factors.append(f"Volatility at {atr_percentile:.0f}th percentile (high uncertainty = opportunity)")
            elif atr_percentile > VOL['atr_mid_percentile']:
                score += VOL['atr_mid_bonus']
            elif atr_percentile < VOL['atr_low_percentile']:
                score -= VOL['atr_low_penalty']
        
        drawdown = indicators.get('drawdown_pct')
        if drawdown:
            if drawdown < VOL['drawdown_severe']:
                score += VOL['drawdown_severe_bonus']
                factors.append(f"Asset down {abs(drawdown):.1f}% from highs (significant dip)")
            elif drawdown < VOL['drawdown_medium']:
                score += VOL['drawdown_medium_bonus']
                factors.append(f"Asset down {abs(drawdown):.1f}% from highs (moderate dip)")
            elif drawdown < VOL['drawdown_light']:
                score += VOL['drawdown_light_bonus']
            elif drawdown > VOL['drawdown_flat']:
                score -= VOL['drawdown_flat_penalty']
        
        return max(0, min(100, score)), factors
    
    @staticmethod
    def calculate_statistical_deviation_score(indicators: Dict) -> Tuple[float, List[str]]:
        factors = []
        score = CFG.score_default_value
        
        z_scores = []
        if indicators.get('z_score_20'):
            z_scores.append(('20-day', indicators['z_score_20']))
        if indicators.get('z_score_50'):
            z_scores.append(('50-day', indicators['z_score_50']))
        if indicators.get('z_score_100'):
            z_scores.append(('100-day', indicators['z_score_100']))
        
        if z_scores:
            avg_z = np.mean([z[1] for z in z_scores])
            
            if avg_z < STAT['extreme_low_z']:
                score += STAT['extreme_low_bonus']
                factors.append(f"Price {abs(avg_z):.1f} std deviations below mean (extreme statistical dip)")
            elif avg_z < STAT['strong_low_z']:
                score += STAT['strong_low_bonus']
                factors.append(f"Price {abs(avg_z):.1f} std deviations below mean (strong statistical dip)")
            elif avg_z < STAT['moderate_low_z']:
                score += STAT['moderate_low_bonus']
                factors.append(f"Price {abs(avg_z):.1f} std deviations below mean (moderate statistical dip)")
            elif avg_z < STAT['light_low_z']:
                score += STAT['light_low_bonus']
            elif avg_z > STAT['high_z']:
                score -= STAT['high_penalty']
            elif avg_z > STAT['moderate_high_z']:
                score -= STAT['moderate_high_penalty']
        
        return max(0, min(100, score)), factors
    
    @staticmethod
    def calculate_macro_fx_score(
        usd_inr_rate: float,
        historical_avg: float = CFG.macro_fx_historical_avg,
    ) -> Tuple[float, List[str]]:
        factors = []
        score = CFG.score_default_value
        
        deviation_pct = ((usd_inr_rate - historical_avg) / historical_avg) * 100
        
        if deviation_pct > FX['high_dev']:
            score -= FX['high_penalty']
            factors.append(f"USD-INR at {usd_inr_rate:.2f}, {deviation_pct:.1f}% above average (expensive for INR buyers)")
        elif deviation_pct > FX['mid_dev']:
            score -= FX['mid_penalty']
        elif deviation_pct < FX['low_dev']:
            score += FX['low_bonus']
            factors.append(f"USD-INR at {usd_inr_rate:.2f}, favorable for INR buyers")
        elif deviation_pct < FX['very_low_dev']:
            score += FX['very_low_bonus']
        
        return max(0, min(100, score)), factors
    
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
