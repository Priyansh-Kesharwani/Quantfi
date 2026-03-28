import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class GeopoliticsConfig:

    g_min: float = 0.8
    g_max: float = 1.2

    source_weights: Dict[str, float] = field(default_factory=lambda: {
        "gdelt": 0.3,
        "newsapi": 0.3,
        "finbert": 0.25,
        "newscatcher": 0.15,
    })

    decay_halflife_days: float = 3.0

    vol_compression_enabled: bool = True
    vol_compression_threshold: float = 0.25                                        

    event_impact: Dict[str, float] = field(default_factory=lambda: {
        "war": -0.15,
        "sanction": -0.10,
        "trade_restriction": -0.08,
        "debt_crisis": -0.12,
        "pandemic": -0.10,
        "political_instability": -0.07,
        "election": 0.0,
        "regulation": -0.03,
        "rate_cut": 0.08,
        "stimulus": 0.10,
        "trade_deal": 0.07,
        "peace_agreement": 0.12,
        "economic_growth": 0.05,
    })

    min_events_for_signal: int = 3

    newsapi_key: Optional[str] = None
    gdelt_enabled: bool = True

    cache_dir: str = "data_cache/geopolitics"
    cache_ttl_hours: int = 6

    seed: int = 42

    def __post_init__(self):
        self.newsapi_key = os.environ.get("NEWSAPI_KEY")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "g_min": self.g_min,
            "g_max": self.g_max,
            "source_weights": self.source_weights,
            "decay_halflife_days": self.decay_halflife_days,
            "vol_compression_enabled": self.vol_compression_enabled,
            "min_events_for_signal": self.min_events_for_signal,
            "seed": self.seed,
        }


@dataclass
class GeoEvent:
    title: str
    event_type: str                                  
    source: str                                                    
    timestamp: datetime
    sentiment_score: float                               
    confidence: float                                   
    affected_assets: List[str] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "event_type": self.event_type,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "sentiment_score": self.sentiment_score,
            "confidence": self.confidence,
            "affected_assets": self.affected_assets,
        }


@dataclass
class GeopoliticsResult:
    G_t: float                                                                    
    raw_sentiment: float                                                        
    n_events: int                                                
    source_breakdown: Dict[str, float]                          
    confidence: float                                                
    is_fallback: bool                                                 
    events_summary: List[Dict[str, Any]]                           
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "G_t": round(self.G_t, 4),
            "raw_sentiment": round(self.raw_sentiment, 4),
            "n_events": self.n_events,
            "source_breakdown": {k: round(v, 4) for k, v in self.source_breakdown.items()},
            "confidence": round(self.confidence, 4),
            "is_fallback": self.is_fallback,
            "events_summary": self.events_summary[:5],
        }


class MockGeoSource:

    name = "mock"

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def fetch_events(
        self,
        symbol: str = "GLOBAL",
        lookback_days: int = 7,
        reference_date: Optional[datetime] = None,
    ) -> List[GeoEvent]:
        ref = reference_date or datetime.utcnow()
        events = []

        date_hash = int(ref.strftime("%Y%m%d")) % 100
        self.rng.seed(date_hash)

        n_events = self.rng.randint(3, 9)

        event_types = list([
            "rate_cut", "rate_cut", "trade_deal", "economic_growth",            
            "election", "regulation",           
            "sanction", "war", "trade_restriction", "political_instability",            
        ])

        for i in range(n_events):
            evt_type = event_types[self.rng.randint(0, len(event_types))]
            days_ago = self.rng.uniform(0, lookback_days)
            sentiment = self.rng.uniform(-0.8, 0.8)

            events.append(GeoEvent(
                title=f"Mock {evt_type.replace('_', ' ').title()} Event #{i+1}",
                event_type=evt_type,
                source="mock",
                timestamp=ref - timedelta(days=days_ago),
                sentiment_score=float(sentiment),
                confidence=float(self.rng.uniform(0.4, 0.9)),
                affected_assets=[symbol] if symbol != "GLOBAL" else ["SPY", "GC=F"],
            ))

        return events


class NewsAPISource:

    name = "newsapi"

    GEOPOLITICAL_KEYWORDS = [
        "sanctions", "tariff", "trade war", "interest rate",
        "central bank", "geopolitical", "conflict", "military",
        "stimulus", "recession", "inflation", "debt crisis",
        "peace", "treaty", "election",
    ]

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("NEWSAPI_KEY")
        self.available = bool(self.api_key)

    def fetch_events(
        self,
        symbol: str = "GLOBAL",
        lookback_days: int = 7,
        reference_date: Optional[datetime] = None,
    ) -> List[GeoEvent]:
        if not self.available:
            logger.debug("NewsAPI key not configured, skipping")
            return []

        ref = reference_date or datetime.utcnow()
        from_date = (ref - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        try:
            from newsapi import NewsApiClient
            newsapi = NewsApiClient(api_key=self.api_key)

            query = " OR ".join(self.GEOPOLITICAL_KEYWORDS[:5])
            articles = newsapi.get_everything(
                q=query,
                from_param=from_date,
                language="en",
                sort_by="relevancy",
                page_size=20,
            )

            events = []
            for article in articles.get("articles", []):
                evt_type = self._classify_event(
                    article.get("title", ""),
                    article.get("description", ""),
                )
                sentiment = self._estimate_sentiment(
                    article.get("title", ""),
                    article.get("description", ""),
                )

                events.append(GeoEvent(
                    title=article.get("title", ""),
                    event_type=evt_type,
                    source="newsapi",
                    timestamp=datetime.fromisoformat(
                        article["publishedAt"].replace("Z", "+00:00")
                    ).replace(tzinfo=None),
                    sentiment_score=sentiment,
                    confidence=0.6,
                    affected_assets=self._extract_assets(article),
                ))

            return events

        except Exception as e:
            logger.warning(f"NewsAPI fetch failed: {e}")
            return []

    def _classify_event(self, title: str, description: str) -> str:
        text = f"{title} {description}".lower()
        if any(w in text for w in ["war", "military", "attack", "conflict"]):
            return "war"
        if any(w in text for w in ["sanction", "embargo"]):
            return "sanction"
        if any(w in text for w in ["tariff", "trade war", "trade restriction"]):
            return "trade_restriction"
        if any(w in text for w in ["rate cut", "dovish", "lower rates"]):
            return "rate_cut"
        if any(w in text for w in ["stimulus", "quantitative easing"]):
            return "stimulus"
        if any(w in text for w in ["peace", "ceasefire", "treaty"]):
            return "peace_agreement"
        if any(w in text for w in ["election", "vote", "ballot"]):
            return "election"
        if any(w in text for w in ["recession", "debt crisis", "default"]):
            return "debt_crisis"
        return "regulation"

    def _estimate_sentiment(self, title: str, description: str) -> float:
        text = f"{title} {description}".lower()
        positive_words = [
            "surge", "rally", "growth", "peace", "stimulus",
            "recovery", "deal", "agreement", "boost", "gain",
        ]
        negative_words = [
            "crash", "war", "crisis", "sanction", "recession",
            "collapse", "plunge", "fear", "threat", "attack",
        ]

        pos_count = sum(1 for w in positive_words if w in text)
        neg_count = sum(1 for w in negative_words if w in text)
        total = pos_count + neg_count

        if total == 0:
            return 0.0
        return float(np.clip((pos_count - neg_count) / total, -1.0, 1.0))

    def _extract_assets(self, article: Dict) -> List[str]:
        text = f"{article.get('title', '')} {article.get('description', '')}".upper()
        known_tickers = ["GOLD", "SILVER", "SPY", "AAPL", "NFLX", "RELIANCE", "TCS"]
        return [t for t in known_tickers if t in text]


class GeopoliticsEngine:

    def __init__(self, config: Optional[GeopoliticsConfig] = None):
        self.config = config or GeopoliticsConfig()

        self.sources: List[Any] = [MockGeoSource(seed=self.config.seed)]

        newsapi_src = NewsAPISource(self.config.newsapi_key)
        if newsapi_src.available:
            self.sources.append(newsapi_src)

        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

    def compute_G_t(
        self,
        symbol: str = "GLOBAL",
        reference_date: Optional[datetime] = None,
        lookback_days: int = 7,
        recent_vol: Optional[float] = None,
    ) -> GeopoliticsResult:
        ref = reference_date or datetime.utcnow()

        all_events: List[GeoEvent] = []
        for source in self.sources:
            try:
                events = source.fetch_events(symbol, lookback_days, ref)
                all_events.extend(events)
            except Exception as e:
                logger.warning(f"Source {source.name} failed: {e}")

        if len(all_events) == 0:
            return self._neutral_fallback("no_events")

        weighted_sum = 0.0
        weight_total = 0.0
        source_scores: Dict[str, List[float]] = {}

        for event in all_events:
            impact = self.config.event_impact.get(event.event_type, 0.0)

            source_w = self.config.source_weights.get(event.source, 0.1)

            age_days = max(0, (ref - event.timestamp).total_seconds() / 86400)
            decay = np.exp(-np.log(2) * age_days / self.config.decay_halflife_days)

            conf = event.confidence

            w = source_w * decay * conf
            contribution = (event.sentiment_score * 0.5 + impact * 0.5)

            weighted_sum += w * contribution
            weight_total += w

            if event.source not in source_scores:
                source_scores[event.source] = []
            source_scores[event.source].append(contribution)

        if weight_total > 0:
            raw_sentiment = weighted_sum / weight_total
        else:
            raw_sentiment = 0.0

        confidence = min(1.0, len(all_events) / max(1, self.config.min_events_for_signal * 3))

        if len(all_events) < self.config.min_events_for_signal:
            confidence *= 0.5
            raw_sentiment *= 0.5                       

        G_t = 1.0 + raw_sentiment
        G_t = float(np.clip(G_t, self.config.g_min, self.config.g_max))

        if (self.config.vol_compression_enabled
                and recent_vol is not None
                and recent_vol > self.config.vol_compression_threshold):
            compression = min(1.0, recent_vol / 0.5)
            G_t = 1.0 + (G_t - 1.0) * (1.0 - compression * 0.5)
            G_t = float(np.clip(G_t, self.config.g_min, self.config.g_max))

        source_breakdown = {
            src: float(np.mean(scores)) if scores else 0.0
            for src, scores in source_scores.items()
        }

        sorted_events = sorted(
            all_events,
            key=lambda e: abs(
                self.config.event_impact.get(e.event_type, 0.0) * e.confidence
            ),
            reverse=True,
        )
        events_summary = [e.to_dict() for e in sorted_events[:5]]

        result = GeopoliticsResult(
            G_t=G_t,
            raw_sentiment=float(raw_sentiment),
            n_events=len(all_events),
            source_breakdown=source_breakdown,
            confidence=float(confidence),
            is_fallback=all(s.name == "mock" for s in self.sources if len(self._get_source_events(all_events, s.name)) > 0),
            events_summary=events_summary,
            meta={
                "symbol": symbol,
                "reference_date": ref.isoformat(),
                "lookback_days": lookback_days,
                "recent_vol": recent_vol,
                "config": self.config.to_dict(),
                "computed_at": datetime.utcnow().isoformat(),
            },
        )

        self._log_computation(result)

        return result

    def compute_G_t_series(
        self,
        dates: pd.DatetimeIndex,
        symbol: str = "GLOBAL",
        recent_vol_series: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        G_t_arr = np.ones(len(dates))

        for i, date in enumerate(dates):
            vol = float(recent_vol_series[i]) if recent_vol_series is not None else None
            try:
                result = self.compute_G_t(
                    symbol=symbol,
                    reference_date=date.to_pydatetime() if hasattr(date, 'to_pydatetime') else date,
                    recent_vol=vol,
                )
                G_t_arr[i] = result.G_t
            except Exception as e:
                logger.warning(f"G_t computation failed for {date}: {e}")
                G_t_arr[i] = 1.0                    

        return G_t_arr

    def _neutral_fallback(self, reason: str) -> GeopoliticsResult:
        return GeopoliticsResult(
            G_t=1.0,
            raw_sentiment=0.0,
            n_events=0,
            source_breakdown={},
            confidence=0.0,
            is_fallback=True,
            events_summary=[],
            meta={"fallback_reason": reason},
        )

    def _get_source_events(self, events: List[GeoEvent], source_name: str) -> List[GeoEvent]:
        return [e for e in events if e.source == source_name]

    def _log_computation(self, result: GeopoliticsResult) -> None:
        try:
            log_path = Path(self.config.cache_dir) / "geopolitics_log.jsonl"
            entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "G_t": result.G_t,
                "n_events": result.n_events,
                "confidence": result.confidence,
                "is_fallback": result.is_fallback,
                "source_breakdown": result.source_breakdown,
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.debug(f"Failed to log G_t: {e}")


def compute_geopolitical_score(
    symbol: str = "GLOBAL",
    config: Optional[GeopoliticsConfig] = None,
    **kwargs,
) -> GeopoliticsResult:
    engine = GeopoliticsEngine(config)
    return engine.compute_G_t(symbol=symbol, **kwargs)


def get_G_t(symbol: str = "GLOBAL", **kwargs) -> float:
    return compute_geopolitical_score(symbol, **kwargs).G_t
