from __future__ import annotations

import os
import re
import html as html_mod
from datetime import datetime, timedelta
from typing import List, Dict, Any
from urllib.parse import quote
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
import logging

from infrastructure.adapters.symbol_resolver import SymbolResolver

logger = logging.getLogger(__name__)


class NewsAdapter:
    def __init__(self, config: Any) -> None:
        self._config = config
        self._client = None
        api_key = os.environ.get("NEWS_API_KEY")
        placeholders = {"placeholder_get_from_newsapi_org", "placeholder", "your_key_here", ""}
        if api_key and api_key not in placeholders:
            try:
                from newsapi import NewsApiClient
                self._client = NewsApiClient(api_key=api_key)
            except ImportError:
                logger.warning("newsapi-python not installed")
        self._default_query = getattr(config, "news_default_query", "gold OR silver OR federal reserve")
        self._page_size = getattr(config, "news_page_size", 20)
        self._lookback_days = getattr(config, "news_lookback_days", 7)
        self._rss_base = getattr(
            config, "news_rss_base",
            "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en",
        )
        self._rss_business = getattr(config, "news_rss_business", "") or ""

    @staticmethod
    def _strip_html(s: str) -> str:
        s = re.sub(r"<[^>]+>", "", s)
        return html_mod.unescape(s).strip()

    def _fetch_rss_news(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        feeds = [self._rss_base.format(query=quote(query))]
        if self._rss_business:
            feeds.append(self._rss_business)
        articles = []
        for url in feeds:
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "Mozilla/5.0"}
                )
                with urllib.request.urlopen(req, timeout=8) as resp:
                    root = ET.fromstring(resp.read())
                for item in root.iter("item"):
                    title = self._strip_html(item.findtext("title", ""))
                    link = item.findtext("link", "")
                    pub = item.findtext("pubDate", "")
                    source_el = item.find("source")
                    source_name = source_el.text if source_el is not None else ""
                    desc = self._strip_html(item.findtext("description", title))
                    if pub:
                        try:
                            pub_dt = parsedate_to_datetime(pub).isoformat()
                        except Exception:
                            pub_dt = pub
                    else:
                        pub_dt = datetime.now().isoformat()
                    articles.append({
                        "title": title,
                        "description": desc,
                        "url": link,
                        "publishedAt": pub_dt,
                        "source": {"name": source_name or "Google News"},
                    })
                if len(articles) >= limit:
                    break
            except Exception as e:
                logger.warning("RSS error (%s...): %s", url[:60], e)
        logger.info("RSS fetched %d articles", len(articles))
        return articles[:limit]

    def fetch_latest_news(
        self,
        query: str = "",
        page_size: int = 20,
    ) -> List[Dict[str, Any]]:
        query = query or self._default_query
        page_size = page_size or self._page_size
        if self._client:
            try:
                from_param = (
                    datetime.now() - timedelta(days=self._lookback_days)
                ).strftime("%Y-%m-%d")
                result = self._client.get_everything(
                    q=query,
                    language="en",
                    sort_by="publishedAt",
                    page_size=page_size,
                    from_param=from_param,
                )
                articles = result.get("articles", [])
                if articles:
                    return articles
            except Exception as e:
                logger.warning("NewsAPI error: %s, falling back to RSS", e)
        return self._fetch_rss_news(query, page_size)

    def fetch_news_for_assets(
        self, symbols: List[str], per_asset: int = 8
    ) -> List[Dict[str, Any]]:
        all_articles = []
        seen_urls = set()
        for sym in symbols:
            keywords = SymbolResolver.get_news_keywords(sym)
            query = " ".join(keywords[:3]) if keywords else sym
            articles = self._fetch_rss_news(query, per_asset)
            for a in articles:
                url = a.get("url", "")
                if url not in seen_urls:
                    seen_urls.add(url)
                    a["_matched_asset"] = sym
                    all_articles.append(a)
        all_articles.sort(key=lambda a: a.get("publishedAt", ""), reverse=True)
        return all_articles
