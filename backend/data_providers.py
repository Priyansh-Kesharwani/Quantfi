import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import os
from newsapi import NewsApiClient
import logging
import time
import random

logger = logging.getLogger(__name__)

# Mock data fallback for when yfinance fails
MOCK_PRICES = {
    'GOLD': {'price': 2050.0, 'high': 2055.0, 'low': 2045.0, 'open': 2048.0, 'volume': 125000},
    'SILVER': {'price': 24.5, 'high': 24.6, 'low': 24.4, 'open': 24.45, 'volume': 85000},
    'GC=F': {'price': 2050.0, 'high': 2055.0, 'low': 2045.0, 'open': 2048.0, 'volume': 125000},
    'SI=F': {'price': 24.5, 'high': 24.6, 'low': 24.4, 'open': 24.45, 'volume': 85000},
    'AAPL': {'price': 185.5, 'high': 186.2, 'low': 184.8, 'open': 185.0, 'volume': 55000000},
    'TSLA': {'price': 245.2, 'high': 247.5, 'low': 243.1, 'open': 244.0, 'volume': 95000000},
    'NFLX': {'price': 725.8, 'high': 730.2, 'low': 721.5, 'open': 724.0, 'volume': 3500000},
    'GOOGL': {'price': 178.5, 'high': 180.1, 'low': 177.2, 'open': 178.0, 'volume': 25000000},
    'MSFT': {'price': 415.2, 'high': 417.5, 'low': 413.0, 'open': 414.5, 'volume': 22000000},
    # Indian stocks (NSE)
    'RELIANCE.NS': {'price': 2925.0, 'high': 2940.0, 'low': 2910.0, 'open': 2920.0, 'volume': 8500000},
    'TCS.NS': {'price': 3850.0, 'high': 3870.0, 'low': 3835.0, 'open': 3845.0, 'volume': 2500000},
    'INFY.NS': {'price': 1580.0, 'high': 1595.0, 'low': 1570.0, 'open': 1575.0, 'volume': 6500000},
    'HDFCBANK.NS': {'price': 1725.0, 'high': 1735.0, 'low': 1715.0, 'open': 1720.0, 'volume': 5500000},
    'ICICIBANK.NS': {'price': 1180.0, 'high': 1190.0, 'low': 1175.0, 'open': 1178.0, 'volume': 7000000},
}

def generate_mock_history(symbol: str, days: int = 730) -> pd.DataFrame:
    """Generate mock historical data for testing"""
    base_price = MOCK_PRICES.get(symbol, MOCK_PRICES.get('GC=F'))['price']
    
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    data = []
    
    current_price = base_price * 0.85  # Start 15% lower
    for date in dates:
        daily_change = random.uniform(-0.02, 0.02)
        current_price *= (1 + daily_change)
        
        data.append({
            'Open': current_price * 0.998,
            'High': current_price * 1.005,
            'Low': current_price * 0.995,
            'Close': current_price,
            'Volume': random.randint(50000, 200000)
        })
    
    df = pd.DataFrame(data, index=dates)
    return df

class PriceProvider:
    """Abstraction for price data - currently using yfinance, easy to swap"""
    
    # Symbol mappings
    SYMBOL_MAP = {
        'GOLD': 'GC=F',  # Gold futures
        'SILVER': 'SI=F',  # Silver futures
        'XAU': 'GC=F',
        'XAG': 'SI=F'
    }
    
    # Indian stock suffix mapping
    INDIAN_EXCHANGES = {
        'NSE': '.NS',
        'BSE': '.BO'
    }
    
    @classmethod
    def get_symbol(cls, asset_symbol: str, exchange: Optional[str] = None) -> str:
        """Convert our symbols to provider symbols"""
        # Handle Indian stocks
        if exchange in cls.INDIAN_EXCHANGES:
            return f"{asset_symbol}{cls.INDIAN_EXCHANGES[exchange]}"
        
        # Handle mapped symbols (metals)
        return cls.SYMBOL_MAP.get(asset_symbol.upper(), asset_symbol)
    
    @classmethod
    def fetch_historical_data(cls, symbol: str, period: str = '2y', exchange: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Fetch historical price data
        period: '1y', '2y', '5y', '10y', 'max'
        """
        try:
            provider_symbol = cls.get_symbol(symbol, exchange)
            ticker = yf.Ticker(provider_symbol)
            df = ticker.history(period=period)
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}, using mock data")
                # Use mock data as fallback
                days_map = {'1mo': 30, '3mo': 90, '6mo': 180, '1y': 365, '2y': 730, '5y': 1825, '10y': 3650, 'max': 7300}
                days = days_map.get(period, 730)
                return generate_mock_history(provider_symbol, days)
            
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}, using mock data")
            # Fallback to mock data
            days_map = {'1mo': 30, '3mo': 90, '6mo': 180, '1y': 365, '2y': 730, '5y': 1825, '10y': 3650, 'max': 7300}
            days = days_map.get(period, 730)
            provider_symbol = cls.get_symbol(symbol, exchange)
            return generate_mock_history(provider_symbol, days)
    
    @classmethod
    def fetch_latest_price(cls, symbol: str, exchange: Optional[str] = None) -> Optional[Dict]:
        """
        Fetch the latest price data
        Returns: {price, volume, timestamp}
        """
        try:
            provider_symbol = cls.get_symbol(symbol, exchange)
            
            # Try yfinance first
            try:
                ticker = yf.Ticker(provider_symbol)
                df = ticker.history(period='1d')
                
                if not df.empty:
                    latest = df.iloc[-1]
                    return {
                        'price': float(latest['Close']),
                        'volume': float(latest['Volume']) if 'Volume' in latest else None,
                        'timestamp': datetime.now(),
                        'high': float(latest['High']),
                        'low': float(latest['Low']),
                        'open': float(latest['Open'])
                    }
            except Exception as yf_error:
                logger.warning(f"yfinance failed for {symbol}: {str(yf_error)}")
            
            # Fallback to mock data
            logger.warning(f"Using mock data for {symbol}")
            if provider_symbol in MOCK_PRICES:
                mock = MOCK_PRICES[provider_symbol]
            elif symbol in MOCK_PRICES:
                mock = MOCK_PRICES[symbol]
            else:
                logger.error(f"No mock data available for {symbol}")
                return None
                
            return {
                'price': mock['price'] * random.uniform(0.998, 1.002),
                'volume': mock['volume'],
                'timestamp': datetime.now(),
                'high': mock['high'],
                'low': mock['low'],
                'open': mock['open']
            }
        except Exception as e:
            logger.error(f"Critical error fetching price for {symbol}: {str(e)}")
            return None

class FXProvider:
    """USD-INR exchange rate provider"""
    
    @classmethod
    def fetch_usd_inr_rate(cls) -> Optional[float]:
        """
        Fetch current USD-INR rate
        """
        try:
            ticker = yf.Ticker('USDINR=X')
            df = ticker.history(period='1d')
            
            if df.empty:
                logger.warning("No USD-INR data, using fallback")
                return 83.5  # Fallback rate
            
            return float(df.iloc[-1]['Close'])
        except Exception as e:
            logger.error(f"Error fetching USD-INR rate: {str(e)}")
            return 83.5  # Fallback

class NewsProvider:
    """News data provider using NewsAPI"""
    
    def __init__(self):
        api_key = os.environ.get('NEWS_API_KEY')
        self.client = NewsApiClient(api_key=api_key) if api_key and api_key != 'placeholder_get_from_newsapi_org' else None
    
    def fetch_latest_news(self, query: str = 'gold OR silver OR federal reserve OR interest rates', page_size: int = 20) -> List[Dict]:
        """
        Fetch latest financial news
        """
        if not self.client:
            logger.warning("News API key not configured")
            return []
        
        try:
            articles = self.client.get_everything(
                q=query,
                language='en',
                sort_by='publishedAt',
                page_size=page_size,
                from_param=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            )
            
            return articles.get('articles', [])
        except Exception as e:
            logger.error(f"Error fetching news: {str(e)}")
            return []
