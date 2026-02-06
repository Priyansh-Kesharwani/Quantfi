#!/usr/bin/env python3

import yfinance as yf
import requests

def test_basic_connectivity():
    """Test basic connectivity and yfinance with simple symbols"""
    
    print("Testing Basic Connectivity and yfinance:")
    print("=" * 50)
    
    # Test basic HTTP connectivity
    try:
        response = requests.get("https://httpbin.org/get", timeout=10)
        print(f"✅ Basic HTTP: {response.status_code}")
    except Exception as e:
        print(f"❌ Basic HTTP: {e}")
    
    # Test yfinance version
    try:
        print(f"📦 yfinance version: {yf.__version__}")
    except:
        print("📦 yfinance version: Unknown")
    
    # Test with very basic US stocks
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']
    
    for symbol in test_symbols:
        try:
            print(f"\n🔍 Testing {symbol}")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period='1d')
            
            if not df.empty:
                price = df.iloc[-1]['Close']
                print(f"✅ {symbol}: ${price:.2f}")
                return True  # If any symbol works, yfinance is functional
            else:
                print(f"❌ {symbol}: No data")
                
        except Exception as e:
            print(f"❌ {symbol}: {str(e)}")
    
    return False

if __name__ == "__main__":
    works = test_basic_connectivity()
    if not works:
        print("\n❌ yfinance appears to be completely non-functional")
    else:
        print("\n✅ yfinance is working for basic symbols")