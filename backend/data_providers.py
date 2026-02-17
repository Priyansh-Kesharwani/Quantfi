import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import os
import re
import logging
import urllib.request
import xml.etree.ElementTree as ET
import html as html_mod
from urllib.parse import quote
from email.utils import parsedate_to_datetime
from app_config import get_backend_config

logger = logging.getLogger(__name__)
CFG = get_backend_config()


class SymbolResolver:

    _info_cache: Dict[str, Dict] = {}

    @classmethod
    def resolve(cls, symbol: str, exchange: Optional[str] = None) -> str:
        aliases = getattr(CFG, 'symbol_aliases', None) or {}
        upper = symbol.upper()
        if upper in aliases:
            return aliases[upper]

        suffixes = getattr(CFG, 'exchange_suffixes', None) or {}
        if exchange and exchange in suffixes:
            sfx = suffixes[exchange]
            return symbol if symbol.endswith(sfx) else f"{symbol}{sfx}"

        return symbol

    @classmethod
    def get_info(cls, symbol: str) -> Dict:
        if symbol in cls._info_cache:
            return cls._info_cache[symbol]
        try:
            info = yf.Ticker(symbol).info or {}
            cls._info_cache[symbol] = info
            return info
        except Exception:
            return {}

    @classmethod
    def get_name(cls, symbol: str) -> str:
        info = cls.get_info(symbol)
        return info.get('longName') or info.get('shortName') or symbol

    @classmethod
    def get_news_keywords(cls, symbol: str) -> List[str]:
        info = cls.get_info(symbol)
        kw = []
        name = info.get('longName') or info.get('shortName') or ''
        if name:
            kw.append(name)
        kw.append(symbol.replace('=F', '').replace('-USD', '').replace('.NS', '').replace('.BO', ''))
        sector = info.get('sector', '')
        if sector:
            kw.append(sector)
        qt = info.get('quoteType', '')
        if qt == 'FUTURE':
            kw.append('futures commodity')
        elif qt == 'CRYPTOCURRENCY':
            kw.append('crypto')
        elif qt == 'ETF':
            kw.append('ETF fund')
        return [k for k in kw if k]


class PriceProvider:

    @classmethod
    def fetch_historical_data(
        cls,
        symbol: str,
        period: str = '2y',
        exchange: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        provider_symbol = SymbolResolver.resolve(symbol, exchange)
        logger.info(f"Fetching data for {provider_symbol} (period={period}, start={start_date}, end={end_date})")

        try:
            ticker = yf.Ticker(provider_symbol)
            if start_date and end_date:
                df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
            else:
                df = ticker.history(period=period, auto_adjust=True)

            if df is None or df.empty:
                logger.error(f"No data for {provider_symbol}")
                return None

            df.columns = [c.title() if c[0].islower() else c for c in df.columns]
            price_cols = [c for c in ['Open', 'High', 'Low', 'Close'] if c in df.columns]
            if price_cols:
                df = df.dropna(subset=price_cols, how='all')

            logger.info(f"Fetched {len(df)} rows for {provider_symbol}: {df.index.min().date()} → {df.index.max().date()}")
            return df
        except Exception as e:
            logger.error(f"Fetch failed for {provider_symbol}: {e}")
            return None

    @classmethod
    def fetch_latest_price(cls, symbol: str, exchange: Optional[str] = None) -> Optional[Dict]:
        try:
            provider_symbol = SymbolResolver.resolve(symbol, exchange)
            ticker = yf.Ticker(provider_symbol)
            df = ticker.history(period=CFG.latest_price_period)

            if df is not None and not df.empty:
                latest = df.iloc[-1]
                return {
                    'price': float(latest['Close']),
                    'volume': float(latest['Volume']) if 'Volume' in latest else None,
                    'timestamp': datetime.now(),
                    'high': float(latest['High']),
                    'low': float(latest['Low']),
                    'open': float(latest['Open']),
                }
            return None
        except Exception as e:
            logger.error(f"Latest price error for {symbol}: {e}")
            return None


class FXProvider:

    @classmethod
    def fetch_usd_inr_rate(cls) -> Optional[float]:
        try:
            ticker = yf.Ticker('USDINR=X')
            df = ticker.history(period=CFG.latest_price_period)
            if df is not None and not df.empty:
                return float(df.iloc[-1]['Close'])
            return CFG.fx_fallback_usd_inr
        except Exception as e:
            logger.error(f"FX rate error: {e}")
            return CFG.fx_fallback_usd_inr


class NewsProvider:

    def __init__(self):
        api_key = os.environ.get('NEWS_API_KEY')
        placeholders = {'placeholder_get_from_newsapi_org', 'placeholder', 'your_key_here', ''}
        self.client = None
        if api_key and api_key not in placeholders:
            try:
                from newsapi import NewsApiClient
                self.client = NewsApiClient(api_key=api_key)
            except ImportError:
                logger.warning("newsapi-python not installed")

    def fetch_latest_news(
        self,
        query: str = CFG.news_default_query,
        page_size: int = CFG.news_page_size,
    ) -> List[Dict]:
        if self.client:
            try:
                articles = self.client.get_everything(
                    q=query,
                    language='en',
                    sort_by='publishedAt',
                    page_size=page_size,
                    from_param=(datetime.now() - timedelta(days=CFG.news_lookback_days)).strftime('%Y-%m-%d'),
                )
                result = articles.get('articles', [])
                if result:
                    return result
            except Exception as e:
                logger.warning(f"NewsAPI error: {e}, falling back to RSS")

        return self._fetch_rss_news(query, page_size)

    def fetch_news_for_assets(self, symbols: List[str], per_asset: int = 8) -> List[Dict]:
        all_articles = []
        seen_urls = set()
        for sym in symbols:
            keywords = SymbolResolver.get_news_keywords(sym)
            query = ' '.join(keywords[:3]) if keywords else sym
            articles = self._fetch_rss_news(query, per_asset)
            for a in articles:
                url = a.get('url', '')
                if url not in seen_urls:
                    seen_urls.add(url)
                    a['_matched_asset'] = sym
                    all_articles.append(a)
        all_articles.sort(key=lambda a: a.get('publishedAt', ''), reverse=True)
        return all_articles

    @staticmethod
    def _strip_html(s: str) -> str:
        s = re.sub(r'<[^>]+>', '', s)
        return html_mod.unescape(s).strip()

    @staticmethod
    def _fetch_rss_news(query: str, limit: int = 20) -> List[Dict]:
        rss_base = getattr(CFG, 'news_rss_base', None) or "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        rss_biz = getattr(CFG, 'news_rss_business', None) or ""

        feeds = [rss_base.format(query=quote(query))]
        if rss_biz:
            feeds.append(rss_biz)

        articles = []
        for url in feeds:
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=8) as resp:
                    root = ET.fromstring(resp.read())
                for item in root.iter('item'):
                    title = NewsProvider._strip_html(item.findtext('title', ''))
                    link = item.findtext('link', '')
                    pub = item.findtext('pubDate', '')
                    source_el = item.find('source')
                    source_name = source_el.text if source_el is not None else ''
                    desc = NewsProvider._strip_html(item.findtext('description', title))

                    if pub:
                        try:
                            pub_dt = parsedate_to_datetime(pub).isoformat()
                        except Exception:
                            pub_dt = pub
                    else:
                        pub_dt = datetime.now().isoformat()

                    articles.append({
                        'title': title,
                        'description': desc,
                        'url': link,
                        'publishedAt': pub_dt,
                        'source': {'name': source_name or 'Google News'},
                    })
                if len(articles) >= limit:
                    break
            except Exception as e:
                logger.warning(f"RSS error ({url[:60]}...): {e}")

        logger.info(f"RSS fetched {len(articles)} articles")
        return articles[:limit]
