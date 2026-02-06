#!/usr/bin/env python3
"""Quick Strategy Verification"""
import sys
sys.path.append('/app/backend')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_providers import PriceProvider
from indicators import TechnicalIndicators
from scoring import ScoringEngine
import json

# Quick test: 3 months, 1 year
ASSETS = [
    {'symbol': 'GOLD', 'exchange': None},
    {'symbol': 'NFLX', 'exchange': None},
]

PERIODS = [('3mo', 90), ('1y', 365)]

def quick_test(asset, period, days):
    """Quick scoring validation"""
    print(f"\n{'='*50}")
    print(f"{asset['symbol']} - {period}")
    print(f"{'='*50}")
    
    df = PriceProvider.fetch_historical_data(asset['symbol'], period, asset['exchange'])
    if df is None or len(df) < 200:
        print(f"❌ Insufficient data")
        return None
    
    # Calculate scores
    scores = []
    for i in range(200, len(df), 5):  # Sample every 5 days
        window = df.iloc[:i+1]
        indicators = TechnicalIndicators.calculate_all_indicators(window)
        
        current_price = float(window.iloc[-1]['Close'])
        score, _, _ = ScoringEngine.calculate_composite_score(indicators, current_price, 83.5)
        
        scores.append({
            'date': window.index[-1],
            'score': score,
            'price': current_price
        })
    
    scores_df = pd.DataFrame(scores)
    
    # Analyze score vs forward returns
    forward_returns = []
    for i in range(len(scores_df) - 10):
        current_score = scores_df.iloc[i]['score']
        current_price = scores_df.iloc[i]['price']
        future_price = scores_df.iloc[i + 10]['price'] if i + 10 < len(scores_df) else scores_df.iloc[-1]['price']
        
        forward_return = ((future_price - current_price) / current_price) * 100
        forward_returns.append({
            'score': current_score,
            'forward_return': forward_return
        })
    
    fr_df = pd.DataFrame(forward_returns)
    
    # Group by score bins
    fr_df['score_bin'] = pd.cut(fr_df['score'], bins=[0, 30, 60, 80, 100], labels=['0-30', '31-60', '61-80', '81-100'])
    grouped = fr_df.groupby('score_bin')['forward_return'].agg(['mean', 'std', 'count'])
    
    print(f"\nScore vs Forward Returns (10-period ahead):")
    print(grouped)
    
    # Check correlation
    corr = fr_df['score'].corr(fr_df['forward_return'])
    print(f"\nCorrelation (score vs forward return): {corr:.3f}")
    
    # Score stats
    print(f"\nScore Distribution:")
    print(f"  Mean: {scores_df['score'].mean():.1f}")
    print(f"  Median: {scores_df['score'].median():.1f}")
    print(f"  Std: {scores_df['score'].std():.1f}")
    
    return {
        'asset': asset['symbol'],
        'period': period,
        'correlation': float(corr),
        'score_stats': {
            'mean': float(scores_df['score'].mean()),
            'median': float(scores_df['score'].median())
        },
        'forward_returns_by_score': grouped.to_dict()
    }

print("Quick Strategy Verification")
print("="*60)

results = []
for asset in ASSETS:
    for period, days in PERIODS:
        result = quick_test(asset, period, days)
        if result:
            results.append(result)

with open('/app/quick_backtest_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to /app/quick_backtest_results.json")
