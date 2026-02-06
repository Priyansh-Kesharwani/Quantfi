#!/usr/bin/env python3

import yfinance as yf
import pandas as pd

def test_gold_symbols():
    """Test different gold symbols to find working ones"""
    
    gold_symbols = [
        ('GC=F', 'Gold Futures (Current)'),
        ('GLD', 'SPDR Gold Trust ETF'),
        ('IAU', 'iShares Gold Trust'),
        ('XAUUSD=X', 'Gold Spot USD'),
        ('GC00', 'Gold Continuous Contract'),
        ('GOLD', 'Barrick Gold Corp'),
        ('/GC', 'Gold Futures Alt'),
        ('XAUUSD', 'Gold USD Alt')
    ]
    
    print("Testing Gold Symbols with yfinance:")
    print("=" * 50)
    
    working_symbols = []
    
    for symbol, description in gold_symbols:
        try:
            print(f"\n🔍 Testing {symbol} ({description})")
            ticker = yf.Ticker(symbol)
            
            # Test 1-day data
            df_1d = ticker.history(period='1d')
            if not df_1d.empty:
                latest_price = df_1d.iloc[-1]['Close']
                print(f"✅ 1-day data: Latest price = ${latest_price:.2f}")
                
                # Test longer period
                df_1y = ticker.history(period='1y')
                if not df_1y.empty:
                    print(f"✅ 1-year data: {len(df_1y)} data points")
                    working_symbols.append((symbol, description, latest_price))
                else:
                    print(f"❌ 1-year data: No data")
            else:
                print(f"❌ 1-day data: No data")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print("\n" + "=" * 50)
    print("WORKING SYMBOLS:")
    for symbol, desc, price in working_symbols:
        print(f"✅ {symbol}: {desc} - ${price:.2f}")
    
    return working_symbols

if __name__ == "__main__":
    working = test_gold_symbols()
    
    if working:
        print(f"\n🎯 Recommended symbol: {working[0][0]} ({working[0][1]})")
    else:
        print("\n❌ No working gold symbols found!")