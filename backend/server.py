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

# Import our modules
from models import (
    Asset, PriceData, IndicatorData, DCAScore, NewsEvent,
    BacktestConfig, BacktestResult, UserSettings
)
from indicators import TechnicalIndicators
from scoring import ScoringEngine
from data_providers import PriceProvider, FXProvider, NewsProvider
from llm_service import LLMService
from backtest import BacktestEngine

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Services
llm_service = LLMService()
news_provider = NewsProvider()

# Create the main app
app = FastAPI(title="Financial Analysis Platform")
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============= REQUEST/RESPONSE MODELS =============

class AddAssetRequest(BaseModel):
    symbol: str
    name: str
    asset_type: str  # 'metal', 'equity', 'indian_equity'
    exchange: Optional[str] = None  # 'NSE', 'BSE', 'NASDAQ', 'NYSE'
    currency: str = 'USD'  # 'USD' or 'INR'

class UpdatePricesRequest(BaseModel):
    symbol: str

class BacktestRequest(BaseModel):
    symbol: str
    start_date: str  # ISO format
    end_date: str  # ISO format
    dca_amount: float
    dca_cadence: str
    buy_dip_threshold: Optional[float] = 60

# ============= ASSET MANAGEMENT =============

@api_router.post("/assets", response_model=Asset)
async def add_asset(request: AddAssetRequest):
    """Add asset to watchlist"""
    # Check if exists
    existing = await db.assets.find_one({'symbol': request.symbol.upper()}, {'_id': 0})
    if existing:
        return Asset(**existing)
    
    asset = Asset(
        symbol=request.symbol.upper(),
        name=request.name,
        asset_type=request.asset_type,
        exchange=request.exchange,
        currency=request.currency
    )
    
    doc = asset.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    await db.assets.insert_one(doc)
    
    # Trigger initial data fetch
    asyncio.create_task(fetch_and_calculate_asset_data(request.symbol.upper(), request.exchange, request.currency))
    
    return asset

@api_router.get("/assets", response_model=List[Asset])
async def get_assets():
    """Get all assets in watchlist"""
    assets = await db.assets.find({'is_active': True}, {'_id': 0}).to_list(100)
    for asset in assets:
        if isinstance(asset['created_at'], str):
            asset['created_at'] = datetime.fromisoformat(asset['created_at'])
    return assets

@api_router.delete("/assets/{symbol}")
async def remove_asset(symbol: str):
    """Remove asset from watchlist"""
    result = await db.assets.update_one(
        {'symbol': symbol.upper()},
        {'$set': {'is_active': False}}
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Asset not found")
    return {"status": "removed"}

# ============= PRICE DATA =============

@api_router.get("/prices/{symbol}")
async def get_latest_price(symbol: str):
    """Get latest price for an asset"""
    # Get asset metadata for exchange info
    asset = await db.assets.find_one({'symbol': symbol.upper()}, {'_id': 0})
    exchange = asset.get('exchange') if asset else None
    currency = asset.get('currency', 'USD') if asset else 'USD'
    
    # Try from cache first
    cached = await db.price_history.find_one(
        {'symbol': symbol.upper()},
        {'_id': 0},
        sort=[('timestamp', -1)]
    )
    
    if cached and isinstance(cached.get('timestamp'), str):
        cached_time = datetime.fromisoformat(cached['timestamp'])
        if datetime.now(timezone.utc) - cached_time.replace(tzinfo=timezone.utc) < timedelta(minutes=5):
            return cached
    
    # Fetch fresh data
    price_data = PriceProvider.fetch_latest_price(symbol.upper(), exchange)
    if not price_data:
        raise HTTPException(status_code=404, detail="Unable to fetch price data")
    
    # Handle currency conversion
    if currency == 'INR':
        # For Indian stocks, price is already in INR
        price_usd = None  # Don't convert
        price_inr = price_data['price']
        usd_inr_rate = None
    else:
        usd_inr_rate = FXProvider.fetch_usd_inr_rate() or 83.5
        price_usd = price_data['price']
        price_inr = price_data['price'] * usd_inr_rate
    
    price_obj = PriceData(
        symbol=symbol.upper(),
        timestamp=price_data['timestamp'],
        price_usd=price_usd or price_inr / (usd_inr_rate or 83.5),  # Fallback conversion
        price_inr=price_inr,
        usd_inr_rate=usd_inr_rate or 83.5,
        volume=price_data.get('volume')
    )
    
    # Cache it
    doc = price_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    await db.price_history.insert_one(doc)
    
    return price_obj

@api_router.get("/prices/{symbol}/history")
async def get_price_history(symbol: str, period: str = "1y"):
    """Get historical price data"""
    df = PriceProvider.fetch_historical_data(symbol.upper(), period)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="No historical data available")
    
    # Convert to JSON format
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

# ============= INDICATORS =============

@api_router.get("/indicators/{symbol}")
async def get_indicators(symbol: str):
    """Get technical indicators for an asset"""
    # Check cache
    cached = await db.indicators.find_one(
        {'symbol': symbol.upper()},
        {'_id': 0},
        sort=[('timestamp', -1)]
    )
    
    if cached and isinstance(cached.get('timestamp'), str):
        cached_time = datetime.fromisoformat(cached['timestamp'])
        if datetime.now(timezone.utc) - cached_time.replace(tzinfo=timezone.utc) < timedelta(hours=1):
            return cached
    
    # Calculate fresh
    df = PriceProvider.fetch_historical_data(symbol.upper(), '2y')
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="Unable to calculate indicators")
    
    indicators = TechnicalIndicators.calculate_all_indicators(df)
    
    indicator_obj = IndicatorData(
        symbol=symbol.upper(),
        timestamp=datetime.now(timezone.utc),
        **indicators
    )
    
    # Cache it
    doc = indicator_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    await db.indicators.insert_one(doc)
    
    return indicator_obj

# ============= DCA SCORING =============

@api_router.get("/scores/{symbol}")
async def get_dca_score(symbol: str):
    """Get DCA favorability score for an asset"""
    # Check cache
    cached = await db.scores.find_one(
        {'symbol': symbol.upper()},
        {'_id': 0},
        sort=[('timestamp', -1)]
    )
    
    if cached and isinstance(cached.get('timestamp'), str):
        cached_time = datetime.fromisoformat(cached['timestamp'])
        if datetime.now(timezone.utc) - cached_time.replace(tzinfo=timezone.utc) < timedelta(hours=1):
            # Parse nested objects
            if 'breakdown' in cached and isinstance(cached['breakdown'], dict):
                return cached
    
    # Calculate fresh score
    # Get indicators
    df = PriceProvider.fetch_historical_data(symbol.upper(), '2y')
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="Unable to calculate score")
    
    indicators = TechnicalIndicators.calculate_all_indicators(df)
    current_price = float(df.iloc[-1]['Close'])
    usd_inr_rate = FXProvider.fetch_usd_inr_rate() or 83.5
    
    # Get user settings for weights
    settings_doc = await db.user_settings.find_one({}, {'_id': 0})
    weights = settings_doc.get('score_weights') if settings_doc else None
    
    # Calculate composite score
    composite_score, breakdown, top_factors = ScoringEngine.calculate_composite_score(
        indicators, current_price, usd_inr_rate, weights
    )
    
    zone = ScoringEngine.get_zone(composite_score)
    
    # Generate explanation using LLM
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
    
    # Cache it
    doc = score_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    doc['breakdown'] = breakdown.model_dump()
    await db.scores.insert_one(doc)
    
    return score_obj

# ============= BACKTESTING =============

@api_router.post("/backtest", response_model=BacktestResult)
async def run_backtest(request: BacktestRequest):
    """Run DCA backtest"""
    try:
        config = BacktestConfig(
            symbol=request.symbol.upper(),
            start_date=datetime.fromisoformat(request.start_date),
            end_date=datetime.fromisoformat(request.end_date),
            dca_amount=request.dca_amount,
            dca_cadence=request.dca_cadence,
            buy_dip_threshold=request.buy_dip_threshold
        )
        
        # Fetch historical data
        period_days = (config.end_date - config.start_date).days
        period = '10y' if period_days > 1825 else '5y' if period_days > 912 else '2y'
        df = PriceProvider.fetch_historical_data(config.symbol, period)
        
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="No historical data available")
        
        # Run backtest
        result = BacktestEngine.run_backtest(config, df)
        
        # Save result
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
        await db.backtest_results.insert_one(doc)
        
        return result
    except Exception as e:
        logger.error(f"Backtest error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============= NEWS & EVENTS =============

@api_router.get("/news")
async def get_news(limit: int = 20):
    """Get latest financial news"""
    # Check cache first
    recent_news = await db.news_events.find(
        {},
        {'_id': 0}
    ).sort('published_at', -1).limit(limit).to_list(limit)
    
    # If cache is recent enough, return it
    if recent_news and len(recent_news) > 0:
        latest = recent_news[0]
        if isinstance(latest.get('published_at'), str):
            latest_time = datetime.fromisoformat(latest['published_at'])
            if datetime.now(timezone.utc) - latest_time.replace(tzinfo=timezone.utc) < timedelta(hours=3):
                return {'news': recent_news}
    
    # Fetch fresh news in background
    asyncio.create_task(fetch_and_classify_news())
    
    return {'news': recent_news}

@api_router.post("/news/refresh")
async def refresh_news():
    """Manually trigger news refresh"""
    await fetch_and_classify_news()
    return {"status": "refreshed"}

# ============= SETTINGS =============

@api_router.get("/settings", response_model=UserSettings)
async def get_settings():
    """Get user settings"""
    settings = await db.user_settings.find_one({}, {'_id': 0})
    if not settings:
        settings = UserSettings().model_dump()
        await db.user_settings.insert_one(settings)
    return UserSettings(**settings)

@api_router.put("/settings", response_model=UserSettings)
async def update_settings(settings: UserSettings):
    """Update user settings"""
    doc = settings.model_dump()
    await db.user_settings.update_one(
        {},
        {'$set': doc},
        upsert=True
    )
    return settings

# ============= DASHBOARD OVERVIEW =============

@api_router.get("/dashboard")
async def get_dashboard():
    """Get complete dashboard data"""
    assets = await db.assets.find({'is_active': True}, {'_id': 0}).to_list(100)
    
    dashboard_data = []
    for asset in assets:
        symbol = asset['symbol']
        
        # Get latest score
        score = await db.scores.find_one(
            {'symbol': symbol},
            {'_id': 0},
            sort=[('timestamp', -1)]
        )
        
        # Get latest price
        price = await db.price_history.find_one(
            {'symbol': symbol},
            {'_id': 0},
            sort=[('timestamp', -1)]
        )
        
        # Get indicators
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

# ============= BACKGROUND TASKS =============

async def fetch_and_calculate_asset_data(symbol: str, exchange: Optional[str] = None, currency: str = 'USD'):
    """Fetch price, calculate indicators and score for an asset"""
    try:
        logger.info(f"Fetching data for {symbol}")
        
        # Fetch price
        price_data = PriceProvider.fetch_latest_price(symbol, exchange)
        if price_data:
            # Handle currency
            if currency == 'INR':
                price_usd = None
                price_inr = price_data['price']
                usd_inr_rate = None
            else:
                usd_inr_rate = FXProvider.fetch_usd_inr_rate() or 83.5
                price_usd = price_data['price']
                price_inr = price_data['price'] * usd_inr_rate
            
            price_obj = PriceData(
                symbol=symbol,
                timestamp=price_data['timestamp'],
                price_usd=price_usd or price_inr / 83.5,
                price_inr=price_inr,
                usd_inr_rate=usd_inr_rate or 83.5,
                volume=price_data.get('volume')
            )
            doc = price_obj.model_dump()
            doc['timestamp'] = doc['timestamp'].isoformat()
            await db.price_history.insert_one(doc)
        
        # Calculate indicators
        df = PriceProvider.fetch_historical_data(symbol, '2y', exchange)
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
            
            # Calculate score
            current_price = float(df.iloc[-1]['Close'])
            usd_inr_rate = FXProvider.fetch_usd_inr_rate() or 83.5
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
    """Fetch and classify news events"""
    try:
        logger.info("Fetching news")
        articles = news_provider.fetch_latest_news()
        
        for article in articles[:10]:  # Limit to avoid rate limits
            # Check if already processed
            existing = await db.news_events.find_one({'url': article['url']})
            if existing:
                continue
            
            # Classify with LLM
            classification = await llm_service.classify_news_event(
                article['title'],
                article['description'] or article['title']
            )
            
            news_obj = NewsEvent(
                title=article['title'],
                description=article['description'] or article['title'],
                source=article['source']['name'],
                url=article['url'],
                published_at=datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00')),
                event_type=classification.get('event_type', 'general'),
                affected_assets=classification.get('affected_assets', []),
                impact_scores=classification.get('impact_scores', {}),
                summary=classification.get('summary', '')
            )
            
            doc = news_obj.model_dump()
            doc['published_at'] = doc['published_at'].isoformat()
            await db.news_events.insert_one(doc)
        
        logger.info("News fetch completed")
    except Exception as e:
        logger.error(f"Error fetching news: {str(e)}")

# ============= HEALTH CHECK =============

@api_router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

# Include the router in the main app
app.include_router(api_router)

# PHASE1: indicator hook - Add Phase 1 development endpoints router here
# from phase1_routes import phase1_router
# app.include_router(phase1_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
