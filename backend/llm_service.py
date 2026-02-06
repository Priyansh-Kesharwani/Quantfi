from emergentintegrations.llm.chat import LlmChat, UserMessage
import os
import json
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class LLMService:
    """LLM service for scoring explanations and news classification"""
    
    def __init__(self):
        self.api_key = os.environ.get('EMERGENT_LLM_KEY')
        
    async def generate_score_explanation(self, symbol: str, score: float, breakdown: Dict, top_factors: List[str]) -> str:
        """
        Generate plain-English explanation of DCA score using GPT-5.2
        """
        try:
            chat = LlmChat(
                api_key=self.api_key,
                session_id=f"score-explain-{symbol}",
                system_message="You are a quantitative financial analyst. Provide concise, clear explanations of DCA scoring for investors. Focus on facts, not advice."
            ).with_model("openai", "gpt-5.2")
            
            prompt = f"""Explain this DCA favorability score for {symbol}:

Composite Score: {score:.1f}/100
Zone: {self._get_zone_name(score)}

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
            return f"Score of {score:.1f}/100 indicates a {self._get_zone_name(score)} opportunity for dollar-cost averaging."
    
    async def classify_news_event(self, title: str, description: str) -> Dict:
        """
        Classify news event using Claude Sonnet 4.5
        Returns: {event_type, affected_assets, impact_scores, summary}
        """
        try:
            chat = LlmChat(
                api_key=self.api_key,
                session_id=f"news-classify",
                system_message="You are a financial news analyst. Classify news events and assess their impact on assets."
            ).with_model("anthropic", "claude-sonnet-4-5-20250929")
            
            prompt = f"""Classify this news event:

Title: {title}
Description: {description}

Respond ONLY with valid JSON in this exact format:
{{
  "event_type": "<one of: rate_change, sanction, trade_restriction, mining_regulation, war, election, general>",
  "affected_assets": ["<list of: GOLD, SILVER, or specific equity symbols>"],
  "impact_scores": {{"<asset>": <confidence 0-1>}},
  "summary": "<one sentence summary>"
}}

Example:
{{
  "event_type": "rate_change",
  "affected_assets": ["GOLD", "SILVER"],
  "impact_scores": {{"GOLD": 0.8, "SILVER": 0.7}},
  "summary": "Fed announces rate cut affecting precious metals."
}}"""
            
            message = UserMessage(text=prompt)
            response = await chat.send_message(message)
            
            # Parse JSON response
            try:
                # Extract JSON from response (may have markdown formatting)
                json_str = response.strip()
                if '```json' in json_str:
                    json_str = json_str.split('```json')[1].split('```')[0].strip()
                elif '```' in json_str:
                    json_str = json_str.split('```')[1].split('```')[0].strip()
                
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON from LLM response: {response}")
                return {
                    'event_type': 'general',
                    'affected_assets': [],
                    'impact_scores': {},
                    'summary': description[:100]
                }
        except Exception as e:
            logger.error(f"Error classifying news: {str(e)}")
            return {
                'event_type': 'general',
                'affected_assets': [],
                'impact_scores': {},
                'summary': description[:100]
            }
    
    @staticmethod
    def _get_zone_name(score: float) -> str:
        if score >= 81:
            return 'strong buy-the-dip'
        elif score >= 61:
            return 'favorable accumulation'
        elif score >= 31:
            return 'neutral'
        else:
            return 'unfavorable'
