from __future__ import annotations

import re
import html as html_mod
from datetime import datetime, timezone
from typing import Any

from domain.protocols import INewsProvider, ILLMService


class NewsClassificationService:
    def __init__(
        self,
        news_provider: INewsProvider,
        llm_service: ILLMService,
        config: Any,
    ) -> None:
        self._news_provider = news_provider
        self._llm_service = llm_service
        self._config = config
        self._process_limit = getattr(config, "news_process_limit", 50)

    @staticmethod
    def _strip_html(s: str) -> str:
        from infrastructure.adapters.utils import strip_html
        return strip_html(s)

    async def fetch_and_classify_news(self, db: Any) -> None:
        from backend.models import NewsEvent

        assets = await db.assets.find(
            {"is_active": True}, {"symbol": 1, "_id": 0}
        ).to_list(100)
        symbols = [a["symbol"] for a in assets]

        if symbols:
            articles = self._news_provider.fetch_news_for_assets(
                symbols, per_asset=5
            )
        else:
            articles = self._news_provider.fetch_latest_news()

        for article in articles[: self._process_limit]:
            url = article.get("url", "")
            if not url:
                continue
            existing = await db.news_events.find_one({"url": url})
            if existing:
                continue

            title = self._strip_html(article.get("title", ""))
            desc = self._strip_html(
                article.get("description", "") or title
            )
            classification = await self._llm_service.classify_news_event(
                title, desc
            )
            matched_asset = article.get("_matched_asset", "")
            affected = classification.get("affected_assets", [])
            if matched_asset and matched_asset not in affected:
                affected.append(matched_asset)

            pub_raw = article.get(
                "publishedAt",
                datetime.now(timezone.utc).isoformat(),
            )
            try:
                pub_dt = datetime.fromisoformat(
                    pub_raw.replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                pub_dt = datetime.now(timezone.utc)

            news_obj = NewsEvent(
                title=title,
                description=desc,
                source=article.get("source", {}).get("name", "Unknown"),
                url=url,
                published_at=pub_dt,
                event_type=classification.get("event_type", "general"),
                affected_assets=affected,
                impact_scores=classification.get("impact_scores", {}),
                summary=classification.get("summary", "") or desc[:200],
            )
            doc = news_obj.model_dump()
            doc["published_at"] = doc["published_at"].isoformat()
            await db.news_events.insert_one(doc)
