from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime, timedelta, timezone
import asyncio

from models import (
    Asset, PriceData, IndicatorData, DCAScore, NewsEvent,
    BacktestConfig, BacktestResult, UserSettings
)
from indicators import TechnicalIndicators
from scoring import ScoringEngine
from data_providers import PriceProvider, FXProvider, NewsProvider
from llm_service import LLMService
from backtest import BacktestEngine
from app_config import get_backend_config

_sentiment_mod = None

def _get_sentiment_mod():
    global _sentiment_mod
    if _sentiment_mod is None:
        import importlib.util as ilu
        import sys as _sys
        sa_path = str(Path(__file__).parent.parent / "indicators" / "sentiment_agent.py")
        spec = ilu.spec_from_file_location("sentiment_agent", sa_path)
        mod = ilu.module_from_spec(spec)
        _sys.modules["sentiment_agent"] = mod
        spec.loader.exec_module(mod)
        _sentiment_mod = mod
    return _sentiment_mod

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

llm_service = LLMService()
news_provider = NewsProvider()

app = FastAPI(title="Financial Analysis Platform")
api_router = APIRouter(prefix="/api")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
CFG = get_backend_config()


class AddAssetRequest(BaseModel):
    symbol: str
    name: str
    asset_type: str                                      
    exchange: Optional[str] = None                                  
    currency: str = 'USD'                  

class UpdatePricesRequest(BaseModel):
    symbol: str

class BacktestRequest(BaseModel):
    symbol: str
    start_date: str              
    end_date: str              
    dca_amount: float
    dca_cadence: str
    buy_dip_threshold: Optional[float] = CFG.backtest_buy_dip_threshold_default


@api_router.post("/assets", response_model=Asset)
async def add_asset(request: AddAssetRequest):
    symbol = request.symbol.upper().strip()
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")

    existing = await db.assets.find_one({'symbol': symbol}, {'_id': 0})
    if existing:
        if existing.get('is_active', True):
            logger.info(f"Asset {symbol} already exists and is active")
            return Asset(**existing)
        else:
            logger.info(f"Reactivating soft-deleted asset {symbol}")
            await db.assets.update_one(
                {'symbol': symbol},
                {'$set': {
                    'is_active': True,
                    'name': request.name or existing.get('name', symbol),
                    'asset_type': request.asset_type or existing.get('asset_type', 'equity'),
                    'exchange': request.exchange or existing.get('exchange'),
                    'currency': request.currency or existing.get('currency', 'USD'),
                }}
            )
            reactivated = await db.assets.find_one({'symbol': symbol}, {'_id': 0})
            if isinstance(reactivated.get('created_at'), str):
                reactivated['created_at'] = datetime.fromisoformat(reactivated['created_at'])
            asyncio.create_task(fetch_and_calculate_asset_data(symbol, request.exchange, request.currency))
            return Asset(**reactivated)
    
    asset = Asset(
        symbol=symbol,
        name=request.name,
        asset_type=request.asset_type,
        exchange=request.exchange,
        currency=request.currency
    )
    
    doc = asset.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    await db.assets.insert_one(doc)
    
    asyncio.create_task(fetch_and_calculate_asset_data(symbol, request.exchange, request.currency))
    
    logger.info(f"Added new asset {symbol} to watchlist")
    return asset

@api_router.get("/assets", response_model=List[Asset])
async def get_assets():
    assets = await db.assets.find({'is_active': True}, {'_id': 0}).to_list(100)
    for asset in assets:
        if isinstance(asset['created_at'], str):
            asset['created_at'] = datetime.fromisoformat(asset['created_at'])
    return assets

@api_router.delete("/assets/{symbol}")
async def remove_asset(symbol: str):
    result = await db.assets.update_one(
        {'symbol': symbol.upper()},
        {'$set': {'is_active': False}}
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Asset not found")
    return {"status": "removed"}


@api_router.get("/prices/{symbol}")
async def get_latest_price(symbol: str):
    asset = await db.assets.find_one({'symbol': symbol.upper()}, {'_id': 0})
    exchange = asset.get('exchange') if asset else None
    currency = asset.get('currency', 'USD') if asset else 'USD'
    
    cached = await db.price_history.find_one(
        {'symbol': symbol.upper()},
        {'_id': 0},
        sort=[('timestamp', -1)]
    )
    
    if cached and isinstance(cached.get('timestamp'), str):
        cached_time = datetime.fromisoformat(cached['timestamp'])
        if datetime.now(timezone.utc) - cached_time.replace(tzinfo=timezone.utc) < timedelta(minutes=CFG.cache_price_minutes):
            return cached
    
    price_data = PriceProvider.fetch_latest_price(symbol.upper(), exchange)
    if not price_data:
        raise HTTPException(status_code=404, detail="Unable to fetch price data")
    
    usd_inr_rate = FXProvider.fetch_usd_inr_rate() or CFG.fx_fallback_usd_inr
    
    if currency == 'INR':
        price_inr = price_data['price']
        price_usd = price_inr / usd_inr_rate
    else:
        price_usd = price_data['price']
        price_inr = price_data['price'] * usd_inr_rate
    
    price_obj = PriceData(
        symbol=symbol.upper(),
        timestamp=price_data['timestamp'],
        price_usd=price_usd,
        price_inr=price_inr,
        usd_inr_rate=usd_inr_rate,
        volume=price_data.get('volume')
    )
    
    doc = price_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    await db.price_history.insert_one(doc)
    
    return price_obj

@api_router.get("/prices/{symbol}/history")
async def get_price_history(symbol: str, period: str = "1y"):
    df = PriceProvider.fetch_historical_data(symbol.upper(), period)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="No historical data available")
    
    history = []
    for date, row in df.iterrows():
        history.append({
            'date': date.isoformat(),
            'open': float(row['Open']),
            'high': float(row['High']),
            'low': float(row['Low']),
            'close': float(row['Close']),
            'volume': float(row['Volume']) if 'Volume' in row else None
        })
    
    return {'symbol': symbol.upper(), 'history': history}


@api_router.get("/indicators/{symbol}")
async def get_indicators(symbol: str):
    cached = await db.indicators.find_one(
        {'symbol': symbol.upper()},
        {'_id': 0},
        sort=[('timestamp', -1)]
    )
    
    if cached and isinstance(cached.get('timestamp'), str):
        cached_time = datetime.fromisoformat(cached['timestamp'])
        if datetime.now(timezone.utc) - cached_time.replace(tzinfo=timezone.utc) < timedelta(hours=CFG.cache_indicators_hours):
            return cached
    
    df = PriceProvider.fetch_historical_data(symbol.upper(), CFG.history_period)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="Unable to calculate indicators")
    
    indicators = TechnicalIndicators.calculate_all_indicators(df)
    
    indicator_obj = IndicatorData(
        symbol=symbol.upper(),
        timestamp=datetime.now(timezone.utc),
        **indicators
    )
    
    doc = indicator_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    await db.indicators.insert_one(doc)
    
    return indicator_obj


@api_router.get("/scores/{symbol}")
async def get_dca_score(symbol: str):
    cached = await db.scores.find_one(
        {'symbol': symbol.upper()},
        {'_id': 0},
        sort=[('timestamp', -1)]
    )
    
    if cached and isinstance(cached.get('timestamp'), str):
        cached_time = datetime.fromisoformat(cached['timestamp'])
        if datetime.now(timezone.utc) - cached_time.replace(tzinfo=timezone.utc) < timedelta(hours=CFG.cache_scores_hours):
            if 'breakdown' in cached and isinstance(cached['breakdown'], dict):
                return cached
    
    df = PriceProvider.fetch_historical_data(symbol.upper(), CFG.history_period)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="Unable to calculate score")
    
    indicators = TechnicalIndicators.calculate_all_indicators(df)
    current_price = float(df.iloc[-1]['Close'])
    usd_inr_rate = FXProvider.fetch_usd_inr_rate() or CFG.fx_fallback_usd_inr
    
    settings_doc = await db.user_settings.find_one({}, {'_id': 0})
    weights = settings_doc.get('score_weights') if settings_doc else None
    
    composite_score, breakdown, top_factors = ScoringEngine.calculate_composite_score(
        indicators, current_price, usd_inr_rate, weights
    )
    
    zone = ScoringEngine.get_zone(composite_score)
    
    explanation = await llm_service.generate_score_explanation(
        symbol.upper(),
        composite_score,
        breakdown.model_dump(),
        top_factors
    )
    
    score_obj = DCAScore(
        symbol=symbol.upper(),
        timestamp=datetime.now(timezone.utc),
        composite_score=composite_score,
        zone=zone,
        breakdown=breakdown,
        explanation=explanation,
        top_factors=top_factors
    )
    
    doc = score_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    doc['breakdown'] = breakdown.model_dump()
    await db.scores.insert_one(doc)
    
    return score_obj


@api_router.post("/backtest", response_model=BacktestResult)
async def run_backtest(request: BacktestRequest):
    try:
        config = BacktestConfig(
            symbol=request.symbol.upper(),
            start_date=datetime.fromisoformat(request.start_date),
            end_date=datetime.fromisoformat(request.end_date),
            dca_amount=request.dca_amount,
            dca_cadence=request.dca_cadence,
            buy_dip_threshold=request.buy_dip_threshold
        )

        fetch_start = config.start_date - timedelta(days=CFG.backtest_padding_days)
        df = PriceProvider.fetch_historical_data(
            config.symbol,
            period='max',
            start_date=fetch_start,
            end_date=config.end_date + timedelta(days=1),
        )

        if df is None or df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No real historical data available for {config.symbol}. "
                       f"Check that the symbol is valid on Yahoo Finance."
            )

        logger.info(
            f"Backtest: {config.symbol} | {len(df)} rows | "
            f"{df.index.min().date()} → {df.index.max().date()}"
        )

        result = BacktestEngine.run_backtest(config, df)

        try:
            doc = result.model_dump()
            doc['created_at'] = doc['created_at'].isoformat()
            doc['config'] = {
                'symbol': config.symbol,
                'start_date': config.start_date.isoformat(),
                'end_date': config.end_date.isoformat(),
                'dca_amount': config.dca_amount,
                'dca_cadence': config.dca_cadence,
                'buy_dip_threshold': config.buy_dip_threshold
            }
            doc.pop('equity_curve', None)
            await db.backtest_results.insert_one(doc)
        except Exception as db_err:
            logger.warning(f"Failed to save backtest to DB: {db_err}")

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Backtest error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/news")
async def get_news(limit: int = 20):
    recent_news = await db.news_events.find(
        {},
        {'_id': 0}
    ).sort('published_at', -1).limit(limit).to_list(limit)
    
    if recent_news and len(recent_news) > 0:
        latest = recent_news[0]
        if isinstance(latest.get('published_at'), str):
            latest_time = datetime.fromisoformat(latest['published_at'])
            if datetime.now(timezone.utc) - latest_time.replace(tzinfo=timezone.utc) < timedelta(hours=CFG.cache_news_hours):
                return {'news': recent_news}
    
    asyncio.create_task(fetch_and_classify_news())
    
    return {'news': recent_news}

@api_router.get("/news/asset/{symbol}")
async def get_news_for_asset(symbol: str, limit: int = 3):
    symbol_upper = symbol.upper()
    
    query = {
        '$or': [
            {'affected_assets': symbol_upper},
            {'title': {'$regex': symbol_upper.replace('.', '\\.'), '$options': 'i'}},
            {'description': {'$regex': symbol_upper.replace('.', '\\.'), '$options': 'i'}},
        ]
    }
    
    results = await db.news_events.find(
        query,
        {'_id': 0}
    ).sort('published_at', -1).limit(limit).to_list(limit)
    
    if not results:
        results = await db.news_events.find(
            {},
            {'_id': 0}
        ).sort('published_at', -1).limit(limit).to_list(limit)
    
    return {'symbol': symbol_upper, 'news': results}

@api_router.post("/news/refresh")
async def refresh_news():
    await fetch_and_classify_news()
    return {"status": "refreshed"}


@api_router.get("/settings", response_model=UserSettings)
async def get_settings():
    settings = await db.user_settings.find_one({}, {'_id': 0})
    if not settings:
        settings = UserSettings().model_dump()
        await db.user_settings.insert_one(settings)
    return UserSettings(**settings)

@api_router.put("/settings", response_model=UserSettings)
async def update_settings(settings: UserSettings):
    doc = settings.model_dump()
    await db.user_settings.update_one(
        {},
        {'$set': doc},
        upsert=True
    )
    return settings


@api_router.get("/dashboard")
async def get_dashboard():
    assets = await db.assets.find({'is_active': True}, {'_id': 0}).to_list(100)
    
    dashboard_data = []
    for asset in assets:
        symbol = asset['symbol']
        
        score = await db.scores.find_one(
            {'symbol': symbol},
            {'_id': 0},
            sort=[('timestamp', -1)]
        )
        
        price = await db.price_history.find_one(
            {'symbol': symbol},
            {'_id': 0},
            sort=[('timestamp', -1)]
        )
        
        indicators = await db.indicators.find_one(
            {'symbol': symbol},
            {'_id': 0},
            sort=[('timestamp', -1)]
        )
        
        dashboard_data.append({
            'asset': asset,
            'score': score,
            'price': price,
            'indicators': indicators
        })
    
    return {'assets': dashboard_data}


async def fetch_and_calculate_asset_data(symbol: str, exchange: Optional[str] = None, currency: str = 'USD'):
    try:
        logger.info(f"Fetching data for {symbol}")
        
        price_data = PriceProvider.fetch_latest_price(symbol, exchange)
        if price_data:
            if currency == 'INR':
                price_usd = None
                price_inr = price_data['price']
                usd_inr_rate = None
            else:
                usd_inr_rate = FXProvider.fetch_usd_inr_rate() or CFG.fx_fallback_usd_inr
                price_usd = price_data['price']
                price_inr = price_data['price'] * usd_inr_rate
            
            price_obj = PriceData(
                symbol=symbol,
                timestamp=price_data['timestamp'],
                price_usd=price_usd or price_inr / CFG.fx_fallback_usd_inr,
                price_inr=price_inr,
                usd_inr_rate=usd_inr_rate or CFG.fx_fallback_usd_inr,
                volume=price_data.get('volume')
            )
            doc = price_obj.model_dump()
            doc['timestamp'] = doc['timestamp'].isoformat()
            await db.price_history.insert_one(doc)
        
        df = PriceProvider.fetch_historical_data(symbol, CFG.history_period, exchange)
        if df is not None and not df.empty:
            indicators = TechnicalIndicators.calculate_all_indicators(df)
            indicator_obj = IndicatorData(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                **indicators
            )
            doc = indicator_obj.model_dump()
            doc['timestamp'] = doc['timestamp'].isoformat()
            await db.indicators.insert_one(doc)
            
            current_price = float(df.iloc[-1]['Close'])
            usd_inr_rate = FXProvider.fetch_usd_inr_rate() or CFG.fx_fallback_usd_inr
            composite_score, breakdown, top_factors = ScoringEngine.calculate_composite_score(
                indicators, current_price, usd_inr_rate
            )
            zone = ScoringEngine.get_zone(composite_score)
            explanation = await llm_service.generate_score_explanation(
                symbol, composite_score, breakdown.model_dump(), top_factors
            )
            
            score_obj = DCAScore(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                composite_score=composite_score,
                zone=zone,
                breakdown=breakdown,
                explanation=explanation,
                top_factors=top_factors
            )
            doc = score_obj.model_dump()
            doc['timestamp'] = doc['timestamp'].isoformat()
            doc['breakdown'] = breakdown.model_dump()
            await db.scores.insert_one(doc)
        
        logger.info(f"Completed data fetch for {symbol}")
    except Exception as e:
        logger.error(f"Error in background task for {symbol}: {str(e)}")

async def fetch_and_classify_news():
    import re
    import html as html_mod
    def _strip_html(s: str) -> str:
        s = re.sub(r'<[^>]+>', '', s)
        return html_mod.unescape(s).strip()

    try:
        logger.info("Fetching news")
        assets = await db.assets.find({'is_active': True}, {'symbol': 1, '_id': 0}).to_list(100)
        symbols = [a['symbol'] for a in assets]

        if symbols:
            articles = news_provider.fetch_news_for_assets(symbols, per_asset=5)
        else:
            articles = news_provider.fetch_latest_news()

        for article in articles[:CFG.news_process_limit]:
            url = article.get('url', '')
            if not url:
                continue
            existing = await db.news_events.find_one({'url': url})
            if existing:
                continue

            title = _strip_html(article.get('title', ''))
            desc = _strip_html(article.get('description', '') or title)

            classification = await llm_service.classify_news_event(title, desc)

            matched_asset = article.get('_matched_asset', '')
            affected = classification.get('affected_assets', [])
            if matched_asset and matched_asset not in affected:
                affected.append(matched_asset)

            pub_raw = article.get('publishedAt', datetime.now(timezone.utc).isoformat())
            try:
                pub_dt = datetime.fromisoformat(pub_raw.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                pub_dt = datetime.now(timezone.utc)

            news_obj = NewsEvent(
                title=title,
                description=desc,
                source=article.get('source', {}).get('name', 'Unknown'),
                url=url,
                published_at=pub_dt,
                event_type=classification.get('event_type', 'general'),
                affected_assets=affected,
                impact_scores=classification.get('impact_scores', {}),
                summary=classification.get('summary', '') or desc[:200],
            )

            doc = news_obj.model_dump()
            doc['published_at'] = doc['published_at'].isoformat()
            await db.news_events.insert_one(doc)

        logger.info(f"News fetch completed, processed {len(articles)} articles")
    except Exception as e:
        logger.error(f"Error fetching news: {str(e)}")


@api_router.get("/sentiment/{symbol}")
async def get_sentiment(symbol: str):
    sym = symbol.upper()
    cached = await db.sentiment.find_one(
        {'symbol': sym}, {'_id': 0}, sort=[('timestamp', -1)]
    )
    if cached:
        ts = cached.get('timestamp', '')
        if isinstance(ts, str):
            try:
                age = datetime.now(timezone.utc) - datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
                if age < timedelta(hours=CFG.cache_scores_hours):
                    return cached
            except Exception:
                pass
        return cached
    return {"symbol": sym, "status": "not_analyzed", "G_t": 1.0, "confidence": 0}


@api_router.post("/sentiment/{symbol}")
async def run_sentiment(symbol: str):
    sym = symbol.upper()
    try:
        sa = _get_sentiment_mod()
        result = await sa.full_sentiment_analysis(sym)
        doc = result.model_dump()
        doc['symbol'] = sym
        doc['timestamp'] = datetime.now(timezone.utc).isoformat()
        doc['top_factors'] = [f.model_dump() for f in result.top_factors]
        doc['citations'] = [c.model_dump() for c in result.citations]
        store_doc = {k: v for k, v in doc.items()}
        await db.sentiment.insert_one(store_doc)
        doc.pop('_id', None)
        return doc
    except Exception as e:
        logger.error(f"Sentiment analysis failed for {sym}: {e}")
        return {
            "symbol": sym, "G_t": 1.0, "raw_sentiment": 0.0,
            "confidence": 0.0, "reasoning": f"Analysis unavailable: {str(e)}",
            "top_factors": [], "citations": [],
        }


@api_router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

app.include_router(api_router)


_cors_raw = os.environ.get('CORS_ORIGINS', '*')
_cors_origins = ['*'] if _cors_raw == '*' else [o.strip() for o in _cors_raw.split(',') if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_credentials=_cors_raw != '*',
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
