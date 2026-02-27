try:
    from emergentintegrations.llm.chat import LlmChat, UserMessage
    HAS_LLM = True
except ImportError:
    HAS_LLM = False

import os
import json
from typing import List, Dict
import logging
from backend.app_config import get_backend_config

logger = logging.getLogger(__name__)
CFG = get_backend_config()

class LLMService:
    
    def __init__(self):
        self.api_key = os.environ.get('EMERGENT_LLM_KEY')
        
    async def generate_score_explanation(self, symbol: str, score: float, breakdown: Dict, top_factors: List[str]) -> str:
        zone = self._get_zone_name(score)
        fallback = self._build_template_explanation(symbol, score, zone, breakdown, top_factors)
        
        if not HAS_LLM or not self.api_key or self.api_key == 'placeholder':
            return fallback
        
        try:
            chat = LlmChat(
                api_key=self.api_key,
                session_id=f"score-explain-{symbol}",
                system_message="You are a quantitative financial analyst. Provide concise, clear explanations of DCA scoring for investors. Focus on facts, not advice."
            ).with_model(CFG.llm_explain_provider, CFG.llm_explain_model)
            
            prompt = f"""Explain this DCA favorability score for {symbol}:

Composite Score: {score:.1f}/100
Zone: {zone}

Breakdown:
- Technical & Momentum: {breakdown['technical_momentum']:.1f}/100
- Volatility & Opportunity: {breakdown['volatility_opportunity']:.1f}/100
- Statistical Deviation: {breakdown['statistical_deviation']:.1f}/100
- Macro & FX: {breakdown['macro_fx']:.1f}/100

Key Factors:
{chr(10).join(f'- {f}' for f in top_factors)}

Provide a 2-3 sentence explanation focusing on what's driving the score and what it means for a DCA investor. Be factual, not advisory."""
            
            message = UserMessage(text=prompt)
            response = await chat.send_message(message)
            
            return response
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return fallback
    
    @staticmethod
    def _build_template_explanation(symbol: str, score: float, zone: str, breakdown: Dict, top_factors: List[str]) -> str:
        parts = [f"DCA score for {symbol} is {score:.1f}/100 ({zone})."]
        
        scores_map = {
            'Technical momentum': breakdown.get('technical_momentum', 50),
            'Volatility opportunity': breakdown.get('volatility_opportunity', 50),
            'Statistical deviation': breakdown.get('statistical_deviation', 50),
            'Macro & FX': breakdown.get('macro_fx', 50),
        }
        best = max(scores_map, key=scores_map.get)
        parts.append(f"The strongest driver is {best} at {scores_map[best]:.0f}/100.")
        
        if top_factors:
            parts.append(f"Key factor: {top_factors[0]}.")
        
        return " ".join(parts)
    
    async def classify_news_event(self, title: str, description: str) -> Dict:
        fallback = {
            'event_type': 'general',
            'affected_assets': [],
            'impact_scores': {},
            'summary': description[:CFG.llm_summary_max_chars]
        }
        if not HAS_LLM or not self.api_key or self.api_key == 'placeholder':
            return fallback
        try:
            chat = LlmChat(
                api_key=self.api_key,
                session_id=f"news-classify",
                system_message="You are a financial news analyst. Classify news events and assess their impact on assets."
            ).with_model(CFG.llm_news_provider, CFG.llm_news_model)
            
            prompt = f"""Classify this news event:

Title: {title}
Description: {description}

Respond ONLY with valid JSON in this exact format:
{{
  "event_type": "<one of: rate_change, sanction, trade_restriction, mining_regulation, war, election, general>",
  "affected_assets": ["<list of affected asset ticker symbols>"],
  "impact_scores": {{"<asset>": <confidence 0-1>}},
  "summary": "<one sentence summary>"
}}

Example:
{{
  "event_type": "rate_change",
  "affected_assets": ["SPY", "TLT"],
  "impact_scores": {{"SPY": 0.8, "TLT": 0.7}},
  "summary": "Central bank policy change affecting broad markets."
}}"""
            
            message = UserMessage(text=prompt)
            response = await chat.send_message(message)
            
            try:
                json_str = response.strip()
                if '```json' in json_str:
                    json_str = json_str.split('```json')[1].split('```')[0].strip()
                elif '```' in json_str:
                    json_str = json_str.split('```')[1].split('```')[0].strip()
                
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON from LLM response: {response}")
                return fallback
        except Exception as e:
            logger.error(f"Error classifying news: {str(e)}")
            return fallback
    
    @staticmethod
    def _get_zone_name(score: float) -> str:
        if score >= CFG.score_zone_strong_buy:
            return 'strong buy-the-dip'
        elif score >= CFG.score_zone_favorable:
            return 'favorable accumulation'
        elif score >= CFG.score_zone_neutral:
            return 'neutral'
        else:
            return 'unfavorable'
