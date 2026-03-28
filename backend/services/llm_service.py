from __future__ import annotations

import json
import logging
import os
from typing import Dict, List

logger = logging.getLogger(__name__)

try:
    from emergentintegrations.llm.chat import LlmChat, UserMessage
    _HAS_LLM_SDK = True
except ImportError:
    _HAS_LLM_SDK = False


def _cfg():
    from backend.core.config import get_backend_config
    return get_backend_config()


class LLMService:

    def __init__(self) -> None:
        self._api_key: str | None = os.environ.get("EMERGENT_LLM_KEY")

    @property
    def _available(self) -> bool:
        return _HAS_LLM_SDK and bool(self._api_key) and self._api_key != "placeholder"

    async def generate_score_explanation(
        self,
        symbol: str,
        composite_score: float,
        breakdown: Dict,
        top_factors: List[str],
    ) -> str:
        cfg = _cfg()
        zone = self._get_zone_name(composite_score)
        fallback = self._build_template_explanation(
            symbol, composite_score, zone, breakdown, top_factors,
        )

        if not self._available:
            return fallback

        try:
            chat = LlmChat(
                api_key=self._api_key,
                session_id=f"score-explain-{symbol}",
                system_message=(
                    "You are a quantitative financial analyst. Provide concise, "
                    "clear explanations of DCA scoring for investors. "
                    "Focus on facts, not advice."
                ),
            ).with_model(cfg.llm_explain_provider, cfg.llm_explain_model)

            prompt = (
                f"Explain this DCA favorability score for {symbol}:\n\n"
                f"Composite Score: {composite_score:.1f}/100\n"
                f"Zone: {zone}\n\n"
                f"Breakdown:\n"
                f"- Technical & Momentum: {breakdown['technical_momentum']:.1f}/100\n"
                f"- Volatility & Opportunity: {breakdown['volatility_opportunity']:.1f}/100\n"
                f"- Statistical Deviation: {breakdown['statistical_deviation']:.1f}/100\n"
                f"- Macro & FX: {breakdown['macro_fx']:.1f}/100\n\n"
                f"Key Factors:\n"
                + "\n".join(f"- {f}" for f in top_factors)
                + "\n\nProvide a 2-3 sentence explanation focusing on what's driving "
                "the score and what it means for a DCA investor. Be factual, not advisory."
            )

            message = UserMessage(text=prompt)
            return await chat.send_message(message)
        except Exception:
            logger.exception("LLM score explanation failed, using template")
            return fallback

    async def classify_news_event(self, title: str, description: str) -> Dict:
        cfg = _cfg()
        fallback = {
            "event_type": "general",
            "affected_assets": [],
            "impact_scores": {},
            "summary": description[: cfg.llm_summary_max_chars],
        }

        if not self._available:
            return fallback

        try:
            chat = LlmChat(
                api_key=self._api_key,
                session_id="news-classify",
                system_message=(
                    "You are a financial news analyst. Classify news events "
                    "and assess their impact on assets."
                ),
            ).with_model(cfg.llm_news_provider, cfg.llm_news_model)

            prompt = (
                f"Classify this news event:\n\n"
                f"Title: {title}\n"
                f"Description: {description}\n\n"
                "Respond ONLY with valid JSON in this exact format:\n"
                "{\n"
                '  "event_type": "<one of: rate_change, sanction, trade_restriction, '
                'mining_regulation, war, election, general>",\n'
                '  "affected_assets": ["<list of affected asset ticker symbols>"],\n'
                '  "impact_scores": {"<asset>": <confidence 0-1>},\n'
                '  "summary": "<one sentence summary>"\n'
                "}"
            )

            message = UserMessage(text=prompt)
            response = await chat.send_message(message)

            json_str = response.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()

            return json.loads(json_str)
        except (json.JSONDecodeError, Exception):
            logger.exception("LLM news classification failed, using fallback")
            return fallback

    @staticmethod
    def _build_template_explanation(
        symbol: str,
        score: float,
        zone: str,
        breakdown: Dict,
        top_factors: List[str],
    ) -> str:
        parts = [f"DCA score for {symbol} is {score:.1f}/100 ({zone})."]

        scores_map = {
            "Technical momentum": breakdown.get("technical_momentum", 50),
            "Volatility opportunity": breakdown.get("volatility_opportunity", 50),
            "Statistical deviation": breakdown.get("statistical_deviation", 50),
            "Macro & FX": breakdown.get("macro_fx", 50),
        }
        best = max(scores_map, key=scores_map.get)
        parts.append(f"The strongest driver is {best} at {scores_map[best]:.0f}/100.")

        if top_factors:
            parts.append(f"Key factor: {top_factors[0]}.")

        return " ".join(parts)

    @staticmethod
    def _get_zone_name(score: float) -> str:
        cfg = _cfg()
        if score >= cfg.score_zone_strong_buy:
            return "strong buy-the-dip"
        elif score >= cfg.score_zone_favorable:
            return "favorable accumulation"
        elif score >= cfg.score_zone_neutral:
            return "neutral"
        return "unfavorable"
