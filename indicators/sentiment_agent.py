from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, field_validator, ConfigDict

logger = logging.getLogger(__name__)

class TierWeights(BaseModel):
    news: float = Field(default=0.45, ge=0.0, le=1.0)
    reddit: float = Field(default=0.25, ge=0.0, le=1.0)
    blogs: float = Field(default=0.30, ge=0.0, le=1.0)

class CollectionLimits(BaseModel):
    max_news_articles: int = Field(default=15, ge=1)
    max_reddit_posts: int = Field(default=5, ge=1)
    max_comments_per_post: int = Field(default=10, ge=1)
    max_reddit_chars: int = Field(default=20_000, ge=1_000)
    max_blog_articles: int = Field(default=5, ge=1)
    max_knowledge_chars: int = Field(default=40_000, ge=10_000)
    news_lookback_days: int = Field(default=7, ge=1)
    reddit_lookback_days: int = Field(default=7, ge=1)
    blog_lookback_days: int = Field(default=14, ge=1)

class LLMConfig(BaseModel):
    model: str = Field(default="groq/llama-3.3-70b-versatile")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=256)
    api_key: Optional[str] = None
    fallback_model: Optional[str] = Field(default="gemini/gemini-2.0-flash-lite")
    timeout: int = Field(default=90, ge=5)

class SentimentAgentConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    llm: LLMConfig = Field(default_factory=LLMConfig)
    tier_weights: TierWeights = Field(default_factory=TierWeights)
    limits: CollectionLimits = Field(default_factory=CollectionLimits)

    g_min: float = Field(default=0.8, ge=0.0, le=1.0)
    g_max: float = Field(default=1.2, ge=1.0, le=2.0)

    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None
    reddit_user_agent: str = "QuantFi-SentimentAgent/2.0"

    newsapi_key: Optional[str] = None

    extra_blog_feeds: Dict[str, str] = Field(default_factory=dict)

    cache_dir: str = "data_cache/sentiment_agent"

    llm_profile_resolution: bool = True

    def resolve_env(self) -> "SentimentAgentConfig":
        if self.llm.api_key is None:
            provider = (self.llm.model.split("/")[0] if "/" in self.llm.model else "").lower()
            key_map = {
                "groq": "GROQ_API_KEY",
                "gemini": "GEMINI_API_KEY",
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
            }
            env_var = key_map.get(provider)
            self.llm.api_key = (
                (os.environ.get(env_var) if env_var else None)
                or os.environ.get("GROQ_API_KEY")
                or os.environ.get("GEMINI_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("EMERGENT_LLM_KEY")
            )
        if self.reddit_client_id is None:
            self.reddit_client_id = os.environ.get("REDDIT_CLIENT_ID")
        if self.reddit_client_secret is None:
            self.reddit_client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
        if self.newsapi_key is None:
            self.newsapi_key = os.environ.get("NEWSAPI_KEY")
        return self

class AssetProfile(BaseModel):
    model_config = ConfigDict(extra="allow")

    symbol: str
    name: str = ""
    quote_type: str = ""                                                                    
    sector: str = ""
    industry: str = ""
    exchange: str = ""
    currency: str = ""
    description: str = ""

    display_name: str = ""
    interpretation_guide: str = ""
    key_factors: List[str] = Field(default_factory=list)
    subreddits: List[str] = Field(default_factory=list)
    news_keywords: List[str] = Field(default_factory=list)
    sentiment_inversion_events: List[str] = Field(default_factory=list)

class SourceDocument(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source_id: str
    tier: str                                                  
    title: str
    content: str
    source_name: str
    url: str = ""
    published_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    upvotes: int = 0
    comment_count: int = 0
    is_comment: bool = False
    parent_id: Optional[str] = None
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)

class KnowledgeSource(BaseModel):
    model_config = ConfigDict(extra="ignore")

    symbol: str
    asset_profile: AssetProfile
    documents: List[SourceDocument] = Field(default_factory=list)
    market_context: Dict[str, Any] = Field(default_factory=dict)
    compilation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_counts: Dict[str, Any] = Field(default_factory=dict)

    def to_prompt_text(self) -> str:
        parts: list[str] = []

        avg_rel = (
            float(np.mean([d.relevance_score for d in self.documents]))
            if self.documents else 0.0
        )
        parts.append(
            f"# KNOWLEDGE SOURCE: {self.symbol} "
            f"({self.asset_profile.display_name})\n"
            f"Compiled: {self.compilation_timestamp.strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"Sources: {self.source_counts}\n"
            f"Average Asset Relevance: {avg_rel:.2f}\n"
        )

        if self.market_context:
            parts.append("## CURRENT MARKET CONTEXT")
            for k, v in self.market_context.items():
                parts.append(f"- {k}: {v}")
            parts.append("")

        news = [d for d in self.documents if d.tier == "news"]
        if news:
            parts.append(f"## TIER 1: VERIFIED NEWS ({len(news)} articles)\n")
            for d in news:
                age_h = max(0, (self.compilation_timestamp - d.published_at).total_seconds() / 3600)
                rel_tag = f"[relevance={d.relevance_score:.2f}]"
                parts.append(
                    f"[{d.source_id}] {rel_tag} **{d.source_name}** — {d.title}\n"
                    f"  ({age_h:.0f}h ago) {d.content}\n"
                    f"  URL: {d.url}\n"
                )

        posts = [d for d in self.documents if d.tier == "reddit" and not d.is_comment]
        if posts:
            parts.append(f"## TIER 2: REDDIT ({len(posts)} posts + threads)\n")
            for p in posts:
                rel_tag = f"[relevance={p.relevance_score:.2f}]"
                parts.append(
                    f"[{p.source_id}] {rel_tag} **{p.source_name}** — {p.title}  "
                    f"(↑{p.upvotes}, {p.comment_count} comments)\n"
                    f"  {p.content}\n"
                )
                comments = [
                    d for d in self.documents
                    if d.is_comment and d.parent_id == p.source_id
                ]
                if comments:
                    parts.append(f"  **Thread ({len(comments)} top comments):**")
                    for c in comments:
                        parts.append(f"    [{c.source_id}] [rel={c.relevance_score:.2f}] (↑{c.upvotes}) {c.content}")
                parts.append("")

        blogs = [d for d in self.documents if d.tier == "blog"]
        if blogs:
            parts.append(f"## TIER 3: EXPERT ANALYSIS ({len(blogs)} articles)\n")
            for d in blogs:
                rel_tag = f"[relevance={d.relevance_score:.2f}]"
                parts.append(
                    f"[{d.source_id}] {rel_tag} **{d.source_name}** — {d.title}\n"
                    f"  {d.content}\n"
                )

        return "\n".join(parts)

    def get_all_source_ids(self) -> List[str]:
        return [d.source_id for d in self.documents]

class SentimentFactor(BaseModel):
    rank: int
    factor: str
    direction: str                                               
    impact_magnitude: float = Field(ge=0.0, le=1.0)
    supporting_sources: List[str] = Field(default_factory=list)
    explanation: str = ""

class SentimentCitation(BaseModel):
    source_id: str
    claim: str
    direction: str

class SentimentResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    G_t: float = Field(ge=0.8, le=1.2)
    raw_sentiment: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    top_factors: List[SentimentFactor] = Field(default_factory=list)
    source_agreement: float = Field(ge=0.0, le=1.0, default=0.5)
    citations: List[SentimentCitation] = Field(default_factory=list)
    reasoning: str = ""
    asset_specific_notes: str = ""
    dispersion_analysis: str = ""
    validation: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)

PROFILE_GENERATION_PROMPT = """You are a financial research assistant.  Given the metadata for a financial asset, generate a sentiment-analysis profile.

Symbol: {symbol}
Name: {name}
Quote type: {quote_type}
Sector: {sector}
Industry: {industry}
Exchange: {exchange}
Currency: {currency}
Description (truncated): {description}

Produce a JSON object with **exactly** these keys:

1. "display_name"  — short human-readable name (e.g. "US Large-Cap Equity", "Gold Futures", "Indian IT Services")
2. "interpretation_guide"  — 3-5 sentences explaining how to interpret sentiment for THIS specific asset.  What events are bullish?  Bearish?  How does it differ from a generic stock?
3. "key_factors"  — JSON array of exactly 5 strings, each naming a macro / geopolitical factor that most affects this asset's price NOW.
4. "subreddits"  — JSON array of 5-8 subreddit names (without r/) most relevant to this asset.
5. "news_keywords"  — JSON array of 10-15 search keywords / phrases for finding news about this asset.
6. "sentiment_inversion_events"  — JSON array of event-type strings (e.g. "war", "sanctions") where sentiment interpretation is INVERTED for this asset compared to a generic equity.  Empty array if none.

Respond with ONLY valid JSON — no markdown fences, no commentary."""

class AssetProfileResolver:

    def __init__(self, config: SentimentAgentConfig):
        self.config = config
        self._cache_path = Path(config.cache_dir) / "profiles"
        self._cache_path.mkdir(parents=True, exist_ok=True)

    async def resolve(self, symbol: str) -> AssetProfile:
        cached = self._load_cache(symbol)
        if cached is not None:
            return cached

        yf_info = await asyncio.to_thread(self._fetch_yfinance_info, symbol)

        profile = AssetProfile(
            symbol=symbol,
            name=yf_info.get("longName") or yf_info.get("shortName") or symbol,
            quote_type=yf_info.get("quoteType", "UNKNOWN"),
            sector=yf_info.get("sector", ""),
            industry=yf_info.get("industry", ""),
            exchange=yf_info.get("exchange", ""),
            currency=yf_info.get("currency", ""),
            description=(yf_info.get("longBusinessSummary") or "")[:500],
        )

        if self.config.llm_profile_resolution:
            profile = await self._enrich_via_llm(profile)
        else:
            profile = self._enrich_heuristic(profile)

        profile = self._ensure_defaults(profile)

        self._save_cache(profile)
        return profile

    @staticmethod
    def _fetch_yfinance_info(symbol: str) -> Dict[str, Any]:
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}
            return info
        except Exception as e:
            logger.warning(f"yfinance lookup failed for {symbol}: {e}")
            return {"shortName": symbol, "quoteType": "UNKNOWN"}

    async def _enrich_via_llm(self, profile: AssetProfile) -> AssetProfile:
        prompt = PROFILE_GENERATION_PROMPT.format(
            symbol=profile.symbol,
            name=profile.name,
            quote_type=profile.quote_type,
            sector=profile.sector,
            industry=profile.industry,
            exchange=profile.exchange,
            currency=profile.currency,
            description=profile.description[:400],
        )

        try:
            raw = await _llm_completion(
                self.config.llm,
                system="You are a financial research assistant. Respond ONLY with valid JSON.",
                user=prompt,
            )
            data = _parse_json_safe(raw)
            if data:
                profile.display_name = data.get("display_name", profile.display_name)
                profile.interpretation_guide = data.get("interpretation_guide", "")
                profile.key_factors = data.get("key_factors", [])[:5]
                profile.subreddits = data.get("subreddits", [])[:8]
                profile.news_keywords = data.get("news_keywords", [])[:15]
                profile.sentiment_inversion_events = data.get("sentiment_inversion_events", [])
        except Exception as e:
            logger.warning(f"LLM profile enrichment failed: {e}")
            profile = self._enrich_heuristic(profile)
        return profile

    @staticmethod
    def _enrich_heuristic(profile: AssetProfile) -> AssetProfile:
        qt = (profile.quote_type or "").upper()
        sym_clean = profile.symbol.upper().replace("=F", "").replace("-USD", "").replace(".NS", "").replace(".BO", "")
        name_lower = (profile.name or "").lower()

        profile.display_name = profile.name or profile.symbol

        if qt == "EQUITY":
            profile.interpretation_guide = (
                f"Analyse sentiment SPECIFICALLY for {profile.name} ({profile.symbol}). "
                f"Sector: {profile.sector or 'N/A'}, Industry: {profile.industry or 'N/A'}. "
                f"Focus on: company earnings, product launches, executive changes, regulatory actions, "
                f"competitive dynamics, and analyst ratings that DIRECTLY affect {profile.name}. "
                f"Sector-wide news should only be considered if it has a DIRECT causal link to {profile.name}."
            )
        elif qt == "CRYPTOCURRENCY":
            profile.interpretation_guide = (
                f"Analyse sentiment for {profile.name} ({sym_clean}). "
                f"Focus on: protocol upgrades, adoption metrics, regulatory actions targeting this specific coin, "
                f"whale movements, exchange listings/delistings, and developer activity. "
                f"General crypto market sentiment is secondary to coin-specific factors."
            )
        elif qt == "FUTURE":
            profile.interpretation_guide = (
                f"Analyse sentiment for {profile.name} ({sym_clean}) commodity futures. "
                f"Focus on: supply/demand fundamentals, inventory reports, producer actions, "
                f"weather/geopolitical events affecting supply chains, and import/export policies. "
                f"General market sentiment is secondary to commodity-specific fundamentals."
            )
        elif qt == "ETF":
            profile.interpretation_guide = (
                f"Analyse sentiment for {profile.name} ({profile.symbol}) ETF. "
                f"Focus on: underlying index/sector performance, fund flows, "
                f"major constituent changes, and sector-specific regulatory or economic events."
            )
        else:
            profile.interpretation_guide = (
                f"Analyse sentiment SPECIFICALLY for {profile.name} ({profile.symbol}). "
                f"Only consider news and discussion that DIRECTLY mentions or impacts this asset."
            )

        asset_specific_subs: list[str] = []
        if qt == "EQUITY":
            simple_name = name_lower.split(",")[0].split(" inc")[0].split(" corp")[0].split(" ltd")[0].strip()
            simple_name_clean = simple_name.replace(" ", "").replace(".", "")
            if simple_name_clean and len(simple_name_clean) > 2:
                asset_specific_subs.append(simple_name_clean)
            if len(sym_clean) <= 5:
                asset_specific_subs.append(sym_clean)
            industry_subs = {
                "Entertainment": ["television", "movies", "entertainment"],
                "Consumer Electronics": ["apple", "technology", "gadgets"],
                "Internet Content & Information": ["technology", "faang"],
                "Software": ["software", "SaaS"],
                "Semiconductors": ["semiconductors", "chipmaking"],
                "Auto Manufacturers": ["electricvehicles", "cars"],
                "Banks": ["banking", "finance"],
                "Oil & Gas": ["energy", "oil"],
                "Biotechnology": ["biotech", "pharma"],
                "Retail": ["retail"],
            }
            industry = profile.industry or ""
            for ind_key, subs in industry_subs.items():
                if ind_key.lower() in industry.lower():
                    asset_specific_subs.extend(subs)
                    break

        type_subs = {
            "CRYPTOCURRENCY": ["cryptocurrency", "CryptoMarkets", "CryptoCurrency", sym_clean.lower()],
            "FUTURE": ["commodities", "Futures", sym_clean.lower()],
            "ETF": ["ETFs", "Bogleheads"],
        }
        base_type_subs = type_subs.get(qt, [])
        all_subs = asset_specific_subs + base_type_subs + ["investing", "StockMarket"]
        seen = set()
        deduped: list[str] = []
        for s in all_subs:
            sl = s.lower()
            if sl not in seen:
                seen.add(sl)
                deduped.append(s)
        profile.subreddits = deduped[:8]

        kw: list[str] = []
        if profile.name:
            kw.append(profile.name)
        kw.append(profile.symbol)
        if sym_clean != profile.symbol:
            kw.append(sym_clean)
        if qt == "EQUITY":
            simple_name = (profile.name or "").split(",")[0].split(" Inc")[0].split(" Corp")[0].split(" Ltd")[0].strip()
            if simple_name and simple_name not in kw:
                kw.append(simple_name)
            kw.extend([f"{simple_name} earnings", f"{simple_name} stock", f"{profile.symbol} stock"])
            if profile.industry:
                kw.append(f"{simple_name} {profile.industry}")
        elif qt == "CRYPTOCURRENCY":
            kw.extend([f"{sym_clean} crypto", f"{sym_clean} blockchain", f"{profile.name} price"])
        elif qt == "FUTURE":
            kw.extend([f"{sym_clean} futures", f"{sym_clean} commodity", f"{sym_clean} supply demand"])
        elif qt == "ETF":
            kw.extend([f"{profile.name} ETF", f"{profile.symbol} fund"])
        profile.news_keywords = kw[:15]

        if not profile.key_factors:
            if qt == "EQUITY":
                profile.key_factors = [
                    f"{profile.name} earnings & revenue growth",
                    f"{profile.industry or 'Industry'} competitive landscape",
                    f"Regulatory environment for {profile.sector or 'sector'}",
                    "Management decisions & strategic direction",
                    "Analyst ratings & institutional sentiment",
                ]
            elif qt == "CRYPTOCURRENCY":
                profile.key_factors = [
                    f"{sym_clean} protocol development & upgrades",
                    "Regulatory actions targeting crypto",
                    f"{sym_clean} adoption & network metrics",
                    "Exchange liquidity & trading volume",
                    "Institutional investment flows",
                ]

        return profile

    @staticmethod
    def _ensure_defaults(profile: AssetProfile) -> AssetProfile:
        if not profile.display_name:
            profile.display_name = profile.name or profile.symbol
        if not profile.interpretation_guide:
            profile.interpretation_guide = (
                f"Analyse sentiment for {profile.symbol} ({profile.name}) "
                f"based on the knowledge source provided."
            )
        if not profile.subreddits:
            profile.subreddits = ["investing", "StockMarket"]
        if not profile.news_keywords:
            profile.news_keywords = [profile.symbol, profile.name or profile.symbol, "market"]
        return profile

    def _load_cache(self, symbol: str) -> Optional[AssetProfile]:
        fp = self._cache_path / f"{_safe_filename(symbol)}.json"
        if fp.exists():
            try:
                age = datetime.utcnow().timestamp() - fp.stat().st_mtime
                if age < 86400:            
                    return AssetProfile.model_validate_json(fp.read_text())
            except Exception:
                pass
        return None

    def _save_cache(self, profile: AssetProfile) -> None:
        try:
            fp = self._cache_path / f"{_safe_filename(profile.symbol)}.json"
            fp.write_text(profile.model_dump_json(indent=2))
        except Exception as e:
            logger.debug(f"Profile cache write failed: {e}")

class NewsCollector:

    def __init__(self, config: SentimentAgentConfig):
        self.api_key = config.newsapi_key
        self.limits = config.limits

    async def collect(self, profile: AssetProfile) -> List[SourceDocument]:
        return await asyncio.to_thread(self._collect_sync, profile)

    def _collect_sync(self, profile: AssetProfile) -> List[SourceDocument]:
        if not self.api_key:
            logger.info("NewsAPI key not set — skipping news tier")
            return []
        try:
            from newsapi import NewsApiClient
            client = NewsApiClient(api_key=self.api_key)
            asset_terms = []
            if profile.name:
                short_name = profile.name.split(",")[0].split(" Inc")[0].split(" Corp")[0].split(" Ltd")[0].strip()
                if short_name:
                    asset_terms.append(f'"{short_name}"')
            asset_terms.append(f'"{profile.symbol}"')
            query = " OR ".join(asset_terms[:3])
            logger.info(f"NewsAPI query for {profile.symbol}: {query}")
            from_date = (
                datetime.utcnow() - timedelta(days=self.limits.news_lookback_days)
            ).strftime("%Y-%m-%d")

            resp = client.get_everything(
                q=query,
                from_param=from_date,
                language="en",
                sort_by="relevancy",
                page_size=self.limits.max_news_articles,
            )
            docs: list[SourceDocument] = []
            for idx, art in enumerate(resp.get("articles", [])[:self.limits.max_news_articles]):
                try:
                    pub = datetime.fromisoformat(
                        art["publishedAt"].replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                except Exception:
                    pub = datetime.utcnow()

                docs.append(SourceDocument(
                    source_id=f"N{idx + 1}",
                    tier="news",
                    title=art.get("title", ""),
                    content=(art.get("description") or art.get("title", ""))[:1500],
                    source_name=art.get("source", {}).get("name", "Unknown"),
                    url=art.get("url", ""),
                    published_at=pub,
                    metadata={"author": art.get("author")},
                ))
            logger.info(f"NewsAPI: collected {len(docs)} articles")
            return docs
        except Exception as e:
            logger.warning(f"NewsAPI collection failed: {e}")
            return []

class RedditCollector:

    def __init__(self, config: SentimentAgentConfig):
        self.config = config
        self.limits = config.limits

    async def collect(self, profile: AssetProfile) -> List[SourceDocument]:
        try:
            import praw  # noqa: F401
            return await asyncio.to_thread(self._collect_praw, profile)
        except ImportError:
            logger.info("PRAW not installed — using Reddit public JSON API")
            return await self._collect_public_json(profile)
        except Exception as e:
            logger.warning(f"PRAW collection failed ({e}), trying public JSON")
            return await self._collect_public_json(profile)

    def _collect_praw(self, profile: AssetProfile) -> List[SourceDocument]:
        import praw

        cid = self.config.reddit_client_id
        csec = self.config.reddit_client_secret
        if not cid or not csec:
            return []

        reddit = praw.Reddit(
            client_id=cid,
            client_secret=csec,
            user_agent=self.config.reddit_user_agent,
        )
        query = self._build_query(profile)
        docs: list[SourceDocument] = []
        post_idx = 0
        total_chars = 0
        time_filter = "week" if self.limits.reddit_lookback_days <= 7 else "month"

        for sub_name in profile.subreddits:
            if post_idx >= self.limits.max_reddit_posts:
                break
            try:
                sub = reddit.subreddit(sub_name)
                for post in sub.search(query, sort="relevance", time_filter=time_filter,
                                       limit=self.limits.max_reddit_posts - post_idx):
                    if post_idx >= self.limits.max_reddit_posts:
                        break
                    post_idx += 1
                    pid = f"R{post_idx}"
                    body = (post.selftext or post.title)[:2000]
                    docs.append(SourceDocument(
                        source_id=pid, tier="reddit", title=post.title,
                        content=body, source_name=f"r/{sub_name}",
                        url=f"https://reddit.com{post.permalink}",
                        published_at=datetime.utcfromtimestamp(post.created_utc),
                        upvotes=post.score, comment_count=post.num_comments,
                        metadata={"subreddit": sub_name,
                                  "upvote_ratio": getattr(post, "upvote_ratio", 0)},
                    ))
                    total_chars += len(body)

                    try:
                        post.comments.replace_more(limit=0)
                        comments = sorted(post.comments.list(),
                                          key=lambda c: getattr(c, "score", 0), reverse=True)
                        for ci, cm in enumerate(comments[:self.limits.max_comments_per_post]):
                            if total_chars >= self.limits.max_reddit_chars:
                                break
                            ctext = (getattr(cm, "body", "") or "")[:500]
                            docs.append(SourceDocument(
                                source_id=f"R{post_idx}.{ci + 1}", tier="reddit",
                                title=f"Comment on: {post.title[:60]}",
                                content=ctext, source_name=f"r/{sub_name}",
                                url=f"https://reddit.com{getattr(cm, 'permalink', '')}",
                                published_at=datetime.utcfromtimestamp(
                                    getattr(cm, "created_utc", datetime.utcnow().timestamp())),
                                upvotes=getattr(cm, "score", 0),
                                is_comment=True, parent_id=pid,
                            ))
                            total_chars += len(ctext)
                    except Exception as ce:
                        logger.debug(f"Comment fetch error: {ce}")
            except Exception as e:
                logger.debug(f"Subreddit r/{sub_name} search error: {e}")
        logger.info(f"Reddit (PRAW): {post_idx} posts, "
                     f"{sum(1 for d in docs if d.is_comment)} comments")
        return docs

    async def _collect_public_json(self, profile: AssetProfile) -> List[SourceDocument]:
        try:
            import aiohttp
        except ImportError:
            logger.warning("aiohttp not available — cannot fetch Reddit")
            return []

        query = self._build_query(profile)
        docs: list[SourceDocument] = []
        post_idx = 0
        total_chars = 0
        headers = {"User-Agent": self.config.reddit_user_agent}

        async with aiohttp.ClientSession(headers=headers) as session:
            for sub in profile.subreddits:
                if post_idx >= self.limits.max_reddit_posts:
                    break
                url = (
                    f"https://www.reddit.com/r/{sub}/search.json"
                    f"?q={query}&restrict_sr=on&sort=relevance&t=week&limit="
                    f"{self.limits.max_reddit_posts - post_idx}"
                )
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                        if resp.status != 200:
                            continue
                        data = await resp.json()
                    for child in data.get("data", {}).get("children", []):
                        if post_idx >= self.limits.max_reddit_posts:
                            break
                        p = child.get("data", {})
                        post_idx += 1
                        pid = f"R{post_idx}"
                        body = (p.get("selftext") or p.get("title", ""))[:2000]
                        docs.append(SourceDocument(
                            source_id=pid, tier="reddit",
                            title=p.get("title", ""),
                            content=body,
                            source_name=f"r/{sub}",
                            url=f"https://reddit.com{p.get('permalink', '')}",
                            published_at=datetime.utcfromtimestamp(p.get("created_utc", 0)),
                            upvotes=p.get("score", 0),
                            comment_count=p.get("num_comments", 0),
                            metadata={"subreddit": sub},
                        ))
                        total_chars += len(body)

                        cmt_url = f"https://www.reddit.com{p.get('permalink', '')}.json?limit=50"
                        try:
                            async with session.get(cmt_url,
                                                   timeout=aiohttp.ClientTimeout(total=10)) as cr:
                                if cr.status == 200:
                                    cmt_data = await cr.json()
                                    comments_raw = self._extract_comments(cmt_data)
                                    comments_raw.sort(key=lambda x: x.get("score", 0), reverse=True)
                                    for ci, cm in enumerate(
                                        comments_raw[:self.limits.max_comments_per_post]
                                    ):
                                        if total_chars >= self.limits.max_reddit_chars:
                                            break
                                        ctext = (cm.get("body") or "")[:500]
                                        docs.append(SourceDocument(
                                            source_id=f"R{post_idx}.{ci + 1}",
                                            tier="reddit",
                                            title=f"Comment on: {p.get('title', '')[:60]}",
                                            content=ctext,
                                            source_name=f"r/{sub}",
                                            url="",
                                            published_at=datetime.utcfromtimestamp(
                                                cm.get("created_utc", 0)),
                                            upvotes=cm.get("score", 0),
                                            is_comment=True,
                                            parent_id=pid,
                                        ))
                                        total_chars += len(ctext)
                        except Exception:
                            pass
                except Exception as e:
                    logger.debug(f"Reddit JSON r/{sub} failed: {e}")

        logger.info(f"Reddit (JSON): {post_idx} posts, "
                     f"{sum(1 for d in docs if d.is_comment)} comments")
        return docs

    @staticmethod
    def _extract_comments(json_data: Any) -> list[dict]:
        out: list[dict] = []
        if not isinstance(json_data, list) or len(json_data) < 2:
            return out
        for child in json_data[1].get("data", {}).get("children", []):
            d = child.get("data", {})
            if d.get("body"):
                out.append(d)
        return out

    @staticmethod
    def _build_query(profile: AssetProfile) -> str:
        """Build a targeted Reddit search query using only asset-specific terms."""
        sym_clean = profile.symbol.replace("=F", "").replace(".NS", "").replace(".BO", "").replace("-USD", "")
        parts = [sym_clean]
        if profile.name:
            short_name = profile.name.split(",")[0].split(" Inc")[0].split(" Corp")[0].split(" Ltd")[0].strip()
            if short_name and short_name.lower() != sym_clean.lower():
                parts.append(short_name)
        return " OR ".join(parts[:3])

class BlogCollector:

    _DEFAULT_FEEDS: Dict[str, str] = {}

    def __init__(self, config: SentimentAgentConfig):
        self.limits = config.limits
        self.feeds: Dict[str, str] = {**self._DEFAULT_FEEDS, **config.extra_blog_feeds}

    async def collect(self, profile: AssetProfile) -> List[SourceDocument]:
        if not self.feeds:
            return []

        try:
            import aiohttp
        except ImportError:
            logger.info("aiohttp not available — skipping blog tier")
            return []

        keywords = {k.lower() for k in profile.news_keywords[:10]}
        cutoff = datetime.utcnow() - timedelta(days=self.limits.blog_lookback_days)

        tasks = [
            self._fetch_feed(name, url, keywords, cutoff)
            for name, url in self.feeds.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        docs: list[SourceDocument] = []
        for r in results:
            if isinstance(r, list):
                docs.extend(r)

        docs = docs[:self.limits.max_blog_articles]
        for i, d in enumerate(docs):
            d.source_id = f"B{i + 1}"
        logger.info(f"Blogs: collected {len(docs)} articles")
        return docs

    @staticmethod
    async def _fetch_feed(
        name: str, url: str,
        keywords: set[str],
        cutoff: datetime,
    ) -> List[SourceDocument]:
        import aiohttp

        docs: list[SourceDocument] = []
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status != 200:
                        return docs
                    text = await resp.text()

            from bs4 import BeautifulSoup
            soup = BeautifulSoup(text, "lxml-xml")                           

            for item in soup.find_all(["item", "entry"]):
                title_tag = item.find("title")
                title = title_tag.get_text(strip=True) if title_tag else ""
                summary_tag = (
                    item.find("description") or item.find("summary") or item.find("content")
                )
                summary = summary_tag.get_text(strip=True)[:2000] if summary_tag else ""
                link_tag = item.find("link")
                link = ""
                if link_tag:
                    link = link_tag.get("href") or link_tag.get_text(strip=True)

                pub_tag = item.find("pubDate") or item.find("published") or item.find("updated")
                pub_date = datetime.utcnow()
                if pub_tag:
                    try:
                        from email.utils import parsedate_to_datetime
                        pub_date = parsedate_to_datetime(pub_tag.get_text(strip=True)).replace(
                            tzinfo=None)
                    except Exception:
                        pass

                if pub_date < cutoff:
                    continue

                combined = f"{title} {summary}".lower()
                if not any(kw in combined for kw in keywords):
                    continue

                docs.append(SourceDocument(
                    source_id="B0",                      
                    tier="blog", title=title, content=summary,
                    source_name=name, url=link, published_at=pub_date,
                ))
        except Exception as e:
            logger.debug(f"Blog feed '{name}' failed: {e}")
        return docs

class AssetRelevanceScorer:
    """
    State-of-the-art multi-signal asset-relevance scoring.

    For each collected document, computes a relevance score [0, 1] using:
      1. Entity Mention Score  — direct mention of symbol/name/products
      2. Sector Proximity Score — same sector/industry context
      3. Causal Chain Score     — describes direct causal impact on the asset
    
    Documents below a configurable threshold are filtered out before
    being sent to the LLM, ensuring only asset-relevant information
    reaches the analysis prompt.
    """

    RELEVANCE_THRESHOLD = 0.15
    COMMENT_RELEVANCE_THRESHOLD = 0.10

    W_ENTITY = 0.60
    W_SECTOR = 0.15
    W_CAUSAL = 0.25

    @classmethod
    def score_documents(
        cls,
        docs: List[SourceDocument],
        profile: AssetProfile,
    ) -> List[SourceDocument]:
        """Score and filter documents for asset relevance."""
        if not docs:
            return []

        entity_terms = cls._build_entity_terms(profile)
        sector_terms = cls._build_sector_terms(profile)

        scored: list[SourceDocument] = []
        for doc in docs:
            text = f"{doc.title} {doc.content}".lower()

            entity_score = cls._entity_mention_score(text, entity_terms)
            sector_score = cls._sector_proximity_score(text, sector_terms)
            causal_score = cls._causal_chain_score(text, entity_terms, profile)

            relevance = (
                cls.W_ENTITY * entity_score +
                cls.W_SECTOR * sector_score +
                cls.W_CAUSAL * causal_score
            )
            doc.relevance_score = round(min(1.0, relevance), 4)

            threshold = (cls.COMMENT_RELEVANCE_THRESHOLD
                         if doc.is_comment else cls.RELEVANCE_THRESHOLD)
            if doc.relevance_score >= threshold:
                scored.append(doc)

        n_filtered = len(docs) - len(scored)
        if n_filtered > 0:
            logger.info(
                f"  relevance filter: kept {len(scored)}/{len(docs)} docs "
                f"(removed {n_filtered} irrelevant)"
            )
        return scored

    @staticmethod
    def _build_entity_terms(profile: AssetProfile) -> List[Tuple[str, float]]:
        """Build weighted entity terms for matching (term, weight)."""
        terms: list[Tuple[str, float]] = []
        sym = profile.symbol.upper()
        sym_clean = sym.replace("=F", "").replace("-USD", "").replace(".NS", "").replace(".BO", "")

        terms.append((sym_clean.lower(), 1.0))
        if sym != sym_clean:
            terms.append((sym.lower(), 1.0))

        if profile.name:
            terms.append((profile.name.lower(), 0.95))
            short = profile.name.split(",")[0].split(" Inc")[0].split(" Corp")[0].split(" Ltd")[0].strip().lower()
            if short and short != profile.name.lower() and len(short) > 2:
                terms.append((short, 0.90))

        if profile.display_name and profile.display_name.lower() not in {t[0] for t in terms}:
            terms.append((profile.display_name.lower(), 0.80))

        return terms

    @staticmethod
    def _build_sector_terms(profile: AssetProfile) -> List[str]:
        """Build sector/industry matching terms."""
        terms: list[str] = []
        if profile.industry:
            terms.append(profile.industry.lower())
        if profile.sector:
            terms.append(profile.sector.lower())
        return terms

    @staticmethod
    def _entity_mention_score(text: str, entity_terms: List[Tuple[str, float]]) -> float:
        """
        Score based on how directly the document mentions the target asset.
        Uses weighted term matching with position bonus (title mentions > body).
        """
        if not entity_terms:
            return 0.0

        max_score = 0.0
        for term, weight in entity_terms:
            if term in text:
                count = text.count(term)
                freq_bonus = 1.0 + min(0.5, (count - 1) * 0.15)
                score = weight * freq_bonus
                max_score = max(max_score, score)

        return min(1.0, max_score)

    @staticmethod
    def _sector_proximity_score(text: str, sector_terms: List[str]) -> float:
        """Score based on sector/industry context match."""
        if not sector_terms:
            return 0.0
        matches = sum(1 for t in sector_terms if t in text)
        return min(1.0, matches / max(1, len(sector_terms)))

    @staticmethod
    def _causal_chain_score(
        text: str,
        entity_terms: List[Tuple[str, float]],
        profile: AssetProfile,
    ) -> float:
        """
        Score based on causal language linking events to the target asset.
        Detects phrases like "could impact Netflix", "affects streaming".
        """
        causal_phrases = [
            "impact on", "affect", "affecting", "affects",
            "could hurt", "could help", "could benefit",
            "threatens", "boosts", "pressures",
            "competitor", "competition from", "competing with",
            "market share", "revenue growth", "earnings impact",
            "subscriber", "customer", "user growth",
        ]

        entity_present = any(term in text for term, _ in entity_terms)
        if not entity_present:
            industry = (profile.industry or "").lower()
            if industry and industry in text:
                return 0.2
            return 0.0

        causal_count = sum(1 for phrase in causal_phrases if phrase in text)
        if causal_count == 0:
            return 0.3
        return min(1.0, 0.4 + causal_count * 0.15)

class SourceQualityRanker:
    """
    Ranks sources by composite quality = credibility × recency × engagement × relevance.

    This implements a state-of-the-art approach:
      - Tier credibility: institutional news (0.9) > expert blogs (0.7) > social (0.5)
      - Recency decay: exponential decay with half-life of 48 hours
      - Engagement signal: normalized upvotes/comments for social sources
      - Relevance: from AssetRelevanceScorer
    """

    TIER_CREDIBILITY = {
        "news": 0.90,
        "blog": 0.70,
        "reddit": 0.50,
    }
    RECENCY_HALF_LIFE_HOURS = 48.0

    @classmethod
    def rank_documents(cls, docs: List[SourceDocument]) -> List[SourceDocument]:
        """Compute quality scores and sort documents by quality descending."""
        now = datetime.utcnow()

        for doc in docs:
            credibility = cls.TIER_CREDIBILITY.get(doc.tier, 0.5)
            recency = cls._recency_score(doc.published_at, now)
            engagement = cls._engagement_score(doc)
            relevance = doc.relevance_score

            doc.quality_score = round(
                0.35 * relevance +
                0.30 * credibility +
                0.20 * recency +
                0.15 * engagement,
                4
            )

        docs.sort(key=lambda d: d.quality_score, reverse=True)
        return docs

    @classmethod
    def _recency_score(cls, published_at: datetime, now: datetime) -> float:
        """Exponential decay: score = 2^(-age_hours / half_life)."""
        age_hours = max(0.0, (now - published_at).total_seconds() / 3600)
        return float(np.power(2, -age_hours / cls.RECENCY_HALF_LIFE_HOURS))

    @staticmethod
    def _engagement_score(doc: SourceDocument) -> float:
        """Normalized engagement from upvotes/comments."""
        if doc.tier != "reddit":
            return 0.5
        if doc.upvotes <= 0:
            return 0.1
        return min(1.0, np.log1p(doc.upvotes) / np.log1p(1000))

class KnowledgeCompiler:
    def __init__(self, config: SentimentAgentConfig):
        self.max_chars = config.limits.max_knowledge_chars

    def compile(
        self,
        profile: AssetProfile,
        news: List[SourceDocument],
        reddit: List[SourceDocument],
        blogs: List[SourceDocument],
        market_context: Dict[str, Any],
    ) -> KnowledgeSource:

        news = AssetRelevanceScorer.score_documents(news, profile)
        reddit = AssetRelevanceScorer.score_documents(reddit, profile)
        blogs = AssetRelevanceScorer.score_documents(blogs, profile)

        all_docs = list(news) + list(reddit) + list(blogs)

        all_docs = SourceQualityRanker.rank_documents(all_docs)

        total = sum(len(d.content) + len(d.title) for d in all_docs)
        if total > self.max_chars:
            all_docs = self._trim(all_docs)

        counts = {
            "news": sum(1 for d in all_docs if d.tier == "news"),
            "reddit_posts": sum(1 for d in all_docs if d.tier == "reddit" and not d.is_comment),
            "reddit_comments": sum(1 for d in all_docs if d.tier == "reddit" and d.is_comment),
            "blogs": sum(1 for d in all_docs if d.tier == "blog"),
            "avg_relevance": round(
                float(np.mean([d.relevance_score for d in all_docs])) if all_docs else 0.0,
                4
            ),
            "avg_quality": round(
                float(np.mean([d.quality_score for d in all_docs])) if all_docs else 0.0,
                4
            ),
        }

        return KnowledgeSource(
            symbol=profile.symbol,
            asset_profile=profile,
            documents=all_docs,
            market_context=market_context,
            compilation_timestamp=datetime.utcnow(),
            source_counts=counts,
        )

    def _trim(self, docs: List[SourceDocument]) -> List[SourceDocument]:
        kept: list[SourceDocument] = []
        chars = 0
        for d in docs:
            c = len(d.content) + len(d.title) + 100
            if chars + c <= self.max_chars:
                kept.append(d)
                chars += c
        return kept

MASTER_ANALYSIS_PROMPT = """You are a quantitative financial sentiment analyst for a DCA investment system.

You compute G_t ∈ [{g_min}, {g_max}] which modifies the composite:
    CompositeScore_t = 100 × clip(0.5 + (RawFavor_t − 0.5) × S × G_t, 0, 1)
G_t > 1.0 → amplify favourable signals.  G_t < 1.0 → dampen (caution).  G_t = 1.0 → neutral.

TARGET ASSET: {symbol} — {display_name}

{interpretation_guide}

Each source has a [relevance=X.XX] tag. This was computed by our asset-relevance scoring system.

**STRICT RULES:**
- ONLY use sources with relevance ≥ 0.30 as PRIMARY evidence for top_factors.
- Sources with relevance < 0.30 may be used ONLY as weak supporting context, never as primary drivers.
- Sources with relevance < 0.15 MUST be COMPLETELY IGNORED — do not cite them.
- If a source discusses a DIFFERENT company/asset (e.g., Microsoft when analysing Netflix), it is IRRELEVANT even if it has a high relevance score. Use your judgment.
- ALL top_factors MUST describe events/conditions that DIRECTLY affect {symbol}. Do NOT list factors about unrelated companies.
- If most sources are low-relevance or about other assets, set confidence LOW (< 0.3) and G_t = 1.0 (neutral).
- NEVER fabricate asset-specific factors from irrelevant sources. If you cannot find enough relevant data, say so honestly.

Tier 1 (Verified News): weight {w_news:.0%} — institutional journalism
Tier 2 (Reddit):        weight {w_reddit:.0%} — crowd sentiment
Tier 3 (Expert Blogs):  weight {w_blogs:.0%} — long-form analysis
When tiers conflict, weight higher tiers more heavily.

1. CITE every factual claim with a source-ID from the knowledge doc ([N1], [R3], [B2]).
2. Track AGREEMENT vs DISAGREEMENT across tiers — dispersion indicates uncertainty → pull G_t toward 1.0.
3. Interpret through THIS asset's profile. The SAME news can be bullish for one asset and bearish for another.
4. Temporal weighting: recent (< 24 h) > older.
5. Conservatism: when uncertain → G_t = 1.0. False signals are worse than no signal.
6. DO NOT PREDICT. G_t describes CURRENT sentiment state.
7. Every factor in top_factors MUST be about {symbol} or directly about its competitive/regulatory environment.
8. In "asset_specific_notes", explain which sources you considered relevant and which you disregarded.

{knowledge_source}

Respond with ONLY this JSON:
{{
  "G_t": <float {g_min}–{g_max}, 4 dp>,
  "raw_sentiment": <float -1.0 to 1.0>,
  "confidence": <float 0–1; LOW if most sources are not about {symbol}>,
  "top_factors": [
    {{"rank":1,"factor":"<MUST be about {symbol}>","direction":"bullish|bearish|neutral","impact_magnitude":<0–1>,"supporting_sources":["N1"],"explanation":"…"}},
    // exactly 5 factors, ALL about {symbol}
  ],
  "source_agreement": <float 0–1>,
  "citations": [{{"source_id":"N1","claim":"…","direction":"bullish|bearish|neutral"}}],
  "reasoning": "2-4 sentences with source-IDs, focused on {symbol}",
  "asset_specific_notes": "Which sources were relevant vs disregarded and why",
  "dispersion_analysis": "1 sentence on cross-tier agreement/disagreement"
}}"""

def build_analysis_prompt(
    knowledge: KnowledgeSource,
    config: SentimentAgentConfig,
) -> str:
    p = knowledge.asset_profile
    return MASTER_ANALYSIS_PROMPT.format(
        g_min=config.g_min,
        g_max=config.g_max,
        symbol=knowledge.symbol,
        display_name=p.display_name,
        interpretation_guide=p.interpretation_guide,
        w_news=config.tier_weights.news,
        w_reddit=config.tier_weights.reddit,
        w_blogs=config.tier_weights.blogs,
        knowledge_source=knowledge.to_prompt_text(),
    )

class PromptValidator:
    REQUIRED = {
        "G_t", "raw_sentiment", "confidence", "top_factors",
        "source_agreement", "citations", "reasoning",
        "asset_specific_notes", "dispersion_analysis",
    }

    def __init__(self, config: SentimentAgentConfig):
        self.g_min = config.g_min
        self.g_max = config.g_max

    def validate(
        self, raw: str, knowledge: KnowledgeSource,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        report: Dict[str, Any] = {"is_valid": True, "errors": [], "warnings": [], "checks": {}}

        parsed = self._parse(raw, report)
        if parsed is None:
            report["is_valid"] = False
            return None, report

        self._check_structure(parsed, report)
        self._check_ranges(parsed, report)
        self._check_citations(parsed, set(knowledge.get_all_source_ids()), report)
        self._check_logic(parsed, report)
        self._check_source_relevance(parsed, knowledge, report)

        avg_cited_rel = report.get("checks", {}).get("cited_avg_relevance", 1.0)
        if parsed and avg_cited_rel < 0.25:
            original_conf = parsed.get("confidence", 0.5)
            penalty = max(0.3, avg_cited_rel / 0.25)
            parsed["confidence"] = round(original_conf * penalty, 4)
            report["checks"]["confidence_penalized"] = True
            report["checks"]["confidence_penalty_factor"] = round(penalty, 4)
            logger.info(
                f"  confidence penalized: {original_conf:.2f} → {parsed['confidence']:.4f} "
                f"(avg cited relevance={avg_cited_rel:.2f})"
            )

        report["is_valid"] = len(report["errors"]) == 0
        return parsed, report

    @staticmethod
    def _parse(raw: str, report: dict) -> Optional[dict]:
        try:
            cleaned = raw.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1].split("```")[0].strip()
            d = json.loads(cleaned)
            report["checks"]["json_valid"] = True
            return d
        except json.JSONDecodeError as e:
            report["checks"]["json_valid"] = False
            report["errors"].append(f"Invalid JSON: {e}")
            return None

    def _check_structure(self, p: dict, r: dict) -> None:
        missing = self.REQUIRED - set(p.keys())
        r["checks"]["structure_missing"] = list(missing)
        if missing:
            r["errors"].append(f"Missing fields: {missing}")

    def _check_ranges(self, p: dict, r: dict) -> None:
        g = p.get("G_t")
        if g is not None and not (self.g_min <= g <= self.g_max):
            r["errors"].append(f"G_t={g} outside [{self.g_min},{self.g_max}]")
        for field, lo, hi in [("raw_sentiment", -1, 1), ("confidence", 0, 1), ("source_agreement", 0, 1)]:
            v = p.get(field)
            if v is not None and not (lo <= v <= hi):
                r["warnings"].append(f"{field}={v} outside [{lo},{hi}]")

    @staticmethod
    def _check_citations(p: dict, valid_ids: set, r: dict) -> None:
        cited = set()
        for c in p.get("citations", []):
            cited.add(c.get("source_id", ""))
        for f in p.get("top_factors", []):
            for s in f.get("supporting_sources", []):
                cited.add(s)
        bad = cited - valid_ids - {""}
        r["checks"]["invalid_citation_ids"] = list(bad)
        if bad:
            r["warnings"].append(f"Non-existent source IDs cited: {bad}")

    @staticmethod
    def _check_logic(p: dict, r: dict) -> None:
        g = p.get("G_t", 1.0)
        factors = p.get("top_factors", [])
        bull = sum(1 for f in factors if f.get("direction") == "bullish")
        bear = sum(1 for f in factors if f.get("direction") == "bearish")
        g_dir = "bullish" if g > 1.02 else ("bearish" if g < 0.98 else "neutral")
        f_dir = "bullish" if bull > bear else ("bearish" if bear > bull else "neutral")
        ok = g_dir == f_dir or "neutral" in (g_dir, f_dir)
        r["checks"]["logical_consistency"] = ok
        if not ok:
            r["warnings"].append(f"G_t direction ({g_dir}) ≠ factor majority ({f_dir})")

    @staticmethod
    def _check_source_relevance(
        p: dict,
        knowledge: "KnowledgeSource",
        r: dict,
    ) -> None:
        """
        Post-analysis validation: check that cited sources have adequate relevance
        to the target asset. Penalize confidence when analysis is based on
        tangentially related or irrelevant sources.
        """
        rel_map: Dict[str, float] = {
            d.source_id: d.relevance_score for d in knowledge.documents
        }

        cited_relevances: list[float] = []
        for factor in p.get("top_factors", []):
            for src_id in factor.get("supporting_sources", []):
                if src_id in rel_map:
                    cited_relevances.append(rel_map[src_id])

        for cit in p.get("citations", []):
            src_id = cit.get("source_id", "")
            if src_id in rel_map:
                cited_relevances.append(rel_map[src_id])

        if not cited_relevances:
            r["checks"]["cited_avg_relevance"] = 0.0
            r["warnings"].append("No cited sources found in knowledge base")
            return

        avg_rel = float(np.mean(cited_relevances))
        low_rel_count = sum(1 for r_val in cited_relevances if r_val < 0.30)
        r["checks"]["cited_avg_relevance"] = round(avg_rel, 4)
        r["checks"]["low_relevance_citations"] = low_rel_count
        r["checks"]["total_citations_checked"] = len(cited_relevances)

        if avg_rel < 0.25:
            r["warnings"].append(
                f"Average cited source relevance is very low ({avg_rel:.2f}). "
                f"Analysis may be based on irrelevant sources."
            )
        if low_rel_count > len(cited_relevances) * 0.5:
            r["warnings"].append(
                f"{low_rel_count}/{len(cited_relevances)} cited sources have "
                f"low relevance (<0.30) to the target asset."
            )

def _resolve_api_key(model: str, explicit_key: Optional[str] = None) -> Optional[str]:
    if explicit_key:
        provider = (model.split("/")[0] if "/" in model else "").lower()
        key_map = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
        env_var = key_map.get(provider)
        if env_var:
            return os.environ.get(env_var) or explicit_key
    return explicit_key

async def _llm_completion(cfg: LLMConfig, *, system: str, user: str) -> str:
    try:
        import litellm
        import ssl
        import certifi
        litellm.drop_params = True
        litellm.ssl_verify = False
        os.environ.setdefault("SSL_VERIFY", "false")
        os.environ.setdefault("LITELLM_SSL_VERIFY", "false")

        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "timeout": cfg.timeout,
        }
        key = _resolve_api_key(cfg.model, cfg.api_key)
        if key:
            kwargs["api_key"] = key

        resp = await litellm.acompletion(**kwargs)
        return resp.choices[0].message.content or ""
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"litellm call failed: {e}")
        if cfg.fallback_model:
            try:
                kwargs["model"] = cfg.fallback_model
                fb_key = _resolve_api_key(cfg.fallback_model)
                if fb_key:
                    kwargs["api_key"] = fb_key
                else:
                    kwargs.pop("api_key", None)
                resp = await litellm.acompletion(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as e2:
                logger.warning(f"litellm fallback failed: {e2}")

    try:
        from emergentintegrations.llm.chat import LlmChat, UserMessage
        provider, model = (cfg.model.split("/", 1) + [""])[:2]
        if not model:
            provider, model = "openai", cfg.model
        api_key = cfg.api_key or os.environ.get("EMERGENT_LLM_KEY", "")
        chat = LlmChat(
            api_key=api_key,
            session_id=f"sa-{hashlib.md5(user[:80].encode()).hexdigest()[:8]}",
            system_message=system,
        ).with_model(provider, model)
        return await chat.send_message(UserMessage(text=user))
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"emergentintegrations call failed: {e}")

    raise RuntimeError("No LLM backend available (install litellm or emergentintegrations)")

class SentimentAgent:

    def __init__(self, config: Optional[SentimentAgentConfig] = None):
        self.config = (config or SentimentAgentConfig()).resolve_env()
        self.profile_resolver = AssetProfileResolver(self.config)
        self.news_collector = NewsCollector(self.config)
        self.reddit_collector = RedditCollector(self.config)
        self.blog_collector = BlogCollector(self.config)
        self.compiler = KnowledgeCompiler(self.config)
        self.validator = PromptValidator(self.config)
        self._log_dir = Path(self.config.cache_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

    async def analyze(
        self,
        symbol: str,
        market_context: Optional[Dict[str, Any]] = None,
    ) -> SentimentResult:
        logger.info(f"[SentimentAgent] analyse({symbol})")

        profile = await self.profile_resolver.resolve(symbol)
        logger.info(f"  profile: {profile.display_name} | type={profile.quote_type}")

        news_task = self.news_collector.collect(profile)
        reddit_task = self.reddit_collector.collect(profile)
        blog_task = self.blog_collector.collect(profile)

        news_docs, reddit_docs, blog_docs = await asyncio.gather(
            news_task, reddit_task, blog_task,
            return_exceptions=False,
        )

        if not isinstance(news_docs, list):
            news_docs = []
        if not isinstance(reddit_docs, list):
            reddit_docs = []
        if not isinstance(blog_docs, list):
            blog_docs = []

        logger.info(f"  collected: news={len(news_docs)}, "
                     f"reddit={len(reddit_docs)}, blogs={len(blog_docs)}")

        ctx = market_context or {"note": "Market context not provided"}
        knowledge = self.compiler.compile(profile, news_docs, reddit_docs, blog_docs, ctx)

        if knowledge.documents:
            avg_rel = float(np.mean([d.relevance_score for d in knowledge.documents]))
            avg_qual = float(np.mean([d.quality_score for d in knowledge.documents]))
            logger.info(
                f"  after relevance filter: {len(knowledge.documents)} docs "
                f"(avg_relevance={avg_rel:.3f}, avg_quality={avg_qual:.3f})"
            )
        else:
            logger.warning(f"  NO relevant documents found for {symbol} — returning neutral")
            return self._neutral(symbol, "no_relevant_sources")

        prompt = build_analysis_prompt(knowledge, self.config)
        logger.info(f"  prompt: {len(prompt)} chars")

        try:
            raw = await _llm_completion(
                self.config.llm,
                system="You are a quantitative financial sentiment analyst. "
                       "Respond ONLY with valid JSON.",
                user=prompt,
            )
        except RuntimeError:
            logger.warning("  LLM unavailable — returning neutral G_t")
            return self._neutral(symbol, "llm_unavailable")

        parsed, report = self.validator.validate(raw, knowledge)
        if parsed is None:
            logger.error(f"  validation FAILED: {report['errors']}")
            return self._neutral(symbol, "validation_failed", report)

        result = SentimentResult(
            G_t=float(np.clip(parsed.get("G_t", 1.0), self.config.g_min, self.config.g_max)),
            raw_sentiment=float(np.clip(parsed.get("raw_sentiment", 0.0), -1, 1)),
            confidence=float(np.clip(parsed.get("confidence", 0.5), 0, 1)),
            top_factors=[SentimentFactor(**f) for f in parsed.get("top_factors", [])[:5]],
            source_agreement=float(np.clip(parsed.get("source_agreement", 0.5), 0, 1)),
            citations=[SentimentCitation(**c) for c in parsed.get("citations", [])],
            reasoning=parsed.get("reasoning", ""),
            asset_specific_notes=parsed.get("asset_specific_notes", ""),
            dispersion_analysis=parsed.get("dispersion_analysis", ""),
            validation=report,
            meta={
                "symbol": symbol,
                "profile_quote_type": profile.quote_type,
                "knowledge_counts": knowledge.source_counts,
                "prompt_chars": len(prompt),
                "llm_model": self.config.llm.model,
                "avg_source_relevance": knowledge.source_counts.get("avg_relevance", 0),
                "avg_source_quality": knowledge.source_counts.get("avg_quality", 0),
            },
        )

        self._log(result)
        logger.info(f"  → G_t={result.G_t:.4f}  conf={result.confidence:.2f}  "
                     f"valid={report['is_valid']}")
        return result

    def _neutral(self, symbol: str, reason: str,
                 report: Optional[Dict] = None) -> SentimentResult:
        return SentimentResult(
            G_t=1.0, raw_sentiment=0.0, confidence=0.0,
            validation=report or {"is_valid": False, "errors": [reason]},
            meta={"symbol": symbol, "fallback_reason": reason},
        )

    def _log(self, result: SentimentResult) -> None:
        try:
            fp = self._log_dir / "sentiment_log.jsonl"
            entry = {
                "ts": datetime.utcnow().isoformat(),
                "symbol": result.meta.get("symbol", ""),
                "G_t": result.G_t,
                "conf": result.confidence,
                "agreement": result.source_agreement,
                "factors": len(result.top_factors),
                "citations": len(result.citations),
                "valid": result.validation.get("is_valid"),
                "avg_relevance": result.meta.get("avg_source_relevance", 0),
                "avg_quality": result.meta.get("avg_source_quality", 0),
                "confidence_penalized": result.validation.get("checks", {}).get("confidence_penalized", False),
            }
            with open(fp, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

async def compute_sentiment_G_t(symbol: str, **kw: Any) -> float:
    agent = SentimentAgent(SentimentAgentConfig(**kw) if kw else None)
    return (await agent.analyze(symbol)).G_t

async def full_sentiment_analysis(symbol: str, **kw: Any) -> SentimentResult:
    agent = SentimentAgent(SentimentAgentConfig(**kw) if kw else None)
    return await agent.analyze(symbol)

def _safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)

def _parse_json_safe(raw: str) -> Optional[dict]:
    try:
        cleaned = raw.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1].split("```")[0].strip()
        return json.loads(cleaned)
    except Exception:
        return None

SentimentBacktestValidator = PromptValidator          
get_asset_profile = None                                    
ASSET_PROFILES = {}                                              
