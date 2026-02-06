from typing import Dict, List, Tuple
import numpy as np
from models import ScoreBreakdown

# PHASE1: indicator hook - Import Phase 1 composite scoring
# from indicators.composite import Phase1CompositeScore
# The Phase 1 composite uses: Gate_t = C_t * L_t * R_t_thresholded
# Opp_t = agg_committee([T_t, U_t * g_pers(H_t)])
# See indicators/composite.py for full implementation

class ScoringEngine:
    """DCA Favorability Scoring Engine"""
    
    # Original weights (momentum-heavy)
    DEFAULT_WEIGHTS = {
        'technical_momentum': 0.4,
        'volatility_opportunity': 0.2,
        'statistical_deviation': 0.2,
        'macro_fx': 0.2
    }
    
    # Tuned weights for buy-low DCA strategy (based on backtest findings)
    TUNED_WEIGHTS = {
        'technical_momentum': 0.25,      # Reduced - avoid trend-chasing
        'volatility_opportunity': 0.25,  # Increased - capitalize on fear
        'statistical_deviation': 0.35,   # Increased - buy statistical dips  
        'macro_fx': 0.15                 # Slight reduction
    }
    
    @staticmethod
    def calculate_technical_momentum_score(indicators: Dict, current_price: float) -> Tuple[float, List[str]]:
        """
        Score based on trend regime and momentum
        Higher score = better buying opportunity
        """
        factors = []
        score = 50.0  # Neutral start
        
        # 200-day SMA regime (30 points)
        if indicators.get('sma_200'):
            if current_price < indicators['sma_200']:
                score += 15
                pct_below = ((indicators['sma_200'] - current_price) / indicators['sma_200']) * 100
                factors.append(f"Price {pct_below:.1f}% below 200-day SMA (buying opportunity)")
            else:
                score -= 5
        
        # RSI (20 points)
        rsi = indicators.get('rsi_14')
        if rsi:
            if rsi < 30:
                score += 20
                factors.append(f"RSI at {rsi:.1f} (oversold territory)")
            elif rsi < 40:
                score += 10
                factors.append(f"RSI at {rsi:.1f} (approaching oversold)")
            elif rsi > 70:
                score -= 15
            elif rsi > 60:
                score -= 5
        
        # MACD histogram slope (15 points)
        macd_hist = indicators.get('macd_hist')
        macd = indicators.get('macd')
        if macd_hist and macd:
            if macd < 0 and macd_hist > 0:
                score += 10
                factors.append("MACD showing bullish divergence")
            elif macd > 0 and macd_hist < 0:
                score -= 10
        
        # Bollinger Band position (15 points)
        bb_lower = indicators.get('bb_lower')
        bb_upper = indicators.get('bb_upper')
        if bb_lower and bb_upper:
            if current_price < bb_lower:
                score += 15
                factors.append("Price below lower Bollinger Band (extreme dip)")
            elif current_price < bb_lower * 1.02:
                score += 8
        
        # ADX trend strength (10 points)
        adx = indicators.get('adx_14')
        if adx:
            if adx < 20:
                score += 5  # Weak trend, good for mean reversion
            elif adx > 40:
                score -= 5  # Strong trend, wait for consolidation
        
        return max(0, min(100, score)), factors
    
    @staticmethod
    def calculate_volatility_opportunity_score(indicators: Dict) -> Tuple[float, List[str]]:
        """
        Score based on volatility and drawdown
        Higher volatility + larger drawdown = better opportunity
        """
        factors = []
        score = 50.0
        
        # ATR percentile (40 points)
        atr_percentile = indicators.get('atr_percentile')
        if atr_percentile:
            if atr_percentile > 80:
                score += 20
                factors.append(f"Volatility at {atr_percentile:.0f}th percentile (high uncertainty = opportunity)")
            elif atr_percentile > 60:
                score += 10
            elif atr_percentile < 30:
                score -= 10
        
        # Drawdown (60 points)
        drawdown = indicators.get('drawdown_pct')
        if drawdown:
            if drawdown < -20:
                score += 30
                factors.append(f"Asset down {abs(drawdown):.1f}% from highs (significant dip)")
            elif drawdown < -10:
                score += 20
                factors.append(f"Asset down {abs(drawdown):.1f}% from highs (moderate dip)")
            elif drawdown < -5:
                score += 10
            elif drawdown > -1:
                score -= 10
        
        return max(0, min(100, score)), factors
    
    @staticmethod
    def calculate_statistical_deviation_score(indicators: Dict) -> Tuple[float, List[str]]:
        """
        Score based on Z-scores (statistical deviation from mean)
        More negative Z-score = better buying opportunity
        """
        factors = []
        score = 50.0
        
        z_scores = []
        if indicators.get('z_score_20'):
            z_scores.append(('20-day', indicators['z_score_20']))
        if indicators.get('z_score_50'):
            z_scores.append(('50-day', indicators['z_score_50']))
        if indicators.get('z_score_100'):
            z_scores.append(('100-day', indicators['z_score_100']))
        
        if z_scores:
            avg_z = np.mean([z[1] for z in z_scores])
            
            if avg_z < -2.0:
                score += 50
                factors.append(f"Price {abs(avg_z):.1f} std deviations below mean (extreme statistical dip)")
            elif avg_z < -1.5:
                score += 35
                factors.append(f"Price {abs(avg_z):.1f} std deviations below mean (strong statistical dip)")
            elif avg_z < -1.0:
                score += 20
                factors.append(f"Price {abs(avg_z):.1f} std deviations below mean (moderate statistical dip)")
            elif avg_z < -0.5:
                score += 10
            elif avg_z > 1.5:
                score -= 30
            elif avg_z > 1.0:
                score -= 15
        
        return max(0, min(100, score)), factors
    
    @staticmethod
    def calculate_macro_fx_score(usd_inr_rate: float, historical_avg: float = 83.0) -> Tuple[float, List[str]]:
        """
        Score based on USD-INR exchange rate
        Higher INR value (lower rate) = better for Indian investors
        """
        factors = []
        score = 50.0
        
        # Compare current rate to historical average
        deviation_pct = ((usd_inr_rate - historical_avg) / historical_avg) * 100
        
        if deviation_pct > 5:
            score -= 20
            factors.append(f"USD-INR at {usd_inr_rate:.2f}, {deviation_pct:.1f}% above average (expensive for INR buyers)")
        elif deviation_pct > 2:
            score -= 10
        elif deviation_pct < -2:
            score += 10
            factors.append(f"USD-INR at {usd_inr_rate:.2f}, favorable for INR buyers")
        elif deviation_pct < -5:
            score += 20
        
        return max(0, min(100, score)), factors
    
    @classmethod
    def calculate_composite_score(
        cls,
        indicators: Dict,
        current_price: float,
        usd_inr_rate: float,
        weights: Dict = None
    ) -> Tuple[float, ScoreBreakdown, List[str]]:
        """
        Calculate composite DCA favorability score
        Returns: (composite_score, breakdown, top_factors)
        """
        if weights is None:
            weights = cls.DEFAULT_WEIGHTS
        
        # Calculate component scores
        tech_score, tech_factors = cls.calculate_technical_momentum_score(indicators, current_price)
        vol_score, vol_factors = cls.calculate_volatility_opportunity_score(indicators)
        stat_score, stat_factors = cls.calculate_statistical_deviation_score(indicators)
        macro_score, macro_factors = cls.calculate_macro_fx_score(usd_inr_rate)
        
        # Weighted composite
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
        
        # Collect all factors and sort by impact
        all_factors = tech_factors + vol_factors + stat_factors + macro_factors
        top_factors = all_factors[:3] if len(all_factors) >= 3 else all_factors
        
        return composite, breakdown, top_factors
    
    @staticmethod
    def get_zone(score: float) -> str:
        """Get zone classification"""
        if score >= 81:
            return 'strong_buy'
        elif score >= 61:
            return 'favorable'
        elif score >= 31:
            return 'neutral'
        else:
            return 'unfavorable'
