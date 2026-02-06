#!/usr/bin/env python3
"""
Comprehensive Backtesting Script
Tests DCA scoring strategy across multiple time horizons
"""
import sys
sys.path.append('/app/backend')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_providers import PriceProvider
from indicators import TechnicalIndicators
from scoring import ScoringEngine
from backtest import BacktestEngine
from models import BacktestConfig
import json

# Test configurations
ASSETS = [
    {'symbol': 'GOLD', 'name': 'Gold', 'exchange': None},
    {'symbol': 'NFLX', 'name': 'Netflix', 'exchange': None},
    {'symbol': 'RELIANCE', 'name': 'Reliance Industries', 'exchange': 'NSE'},
    {'symbol': 'TCS', 'name': 'TCS', 'exchange': 'NSE'},
]

TIME_HORIZONS = [
    ('1mo', 1),
    ('3mo', 3),
    ('6mo', 6),
    ('1y', 12),
    ('3y', 36),
    ('5y', 60),
    ('10y', 120),
]

DCA_AMOUNT = 5000  # INR
USD_INR_RATE = 83.5

def calculate_metrics(returns_series, initial_investment):
    """Calculate performance metrics"""
    if len(returns_series) == 0 or initial_investment == 0:
        return {}
    
    total_return = ((returns_series.iloc[-1] - initial_investment) / initial_investment) * 100
    
    # Calculate drawdown
    cumulative = returns_series
    running_max = cumulative.expanding().max()
    drawdown = ((cumulative - running_max) / running_max) * 100
    max_drawdown = drawdown.min()
    
    # Volatility (annualized)
    returns_pct = returns_series.pct_change().dropna()
    volatility = returns_pct.std() * np.sqrt(252) * 100 if len(returns_pct) > 1 else 0
    
    # CAGR
    years = len(returns_series) / 252  # Approximate trading days
    if years > 0 and returns_series.iloc[-1] > 0 and initial_investment > 0:
        cagr = ((returns_series.iloc[-1] / initial_investment) ** (1/years) - 1) * 100
    else:
        cagr = 0
    
    # Sharpe (simplified, assuming 0 risk-free rate)
    sharpe = (returns_pct.mean() / returns_pct.std() * np.sqrt(252)) if returns_pct.std() > 0 else 0
    
    return {
        'total_return_pct': total_return,
        'cagr': cagr,
        'max_drawdown': max_drawdown,
        'volatility': volatility,
        'sharpe_ratio': sharpe
    }

def backtest_asset(asset, period, months_back):
    """Backtest a single asset over a period"""
    print(f"\n{'='*60}")
    print(f"Testing {asset['name']} ({asset['symbol']}) - {period} period")
    print(f"{'='*60}")
    
    try:
        # Fetch historical data
        df = PriceProvider.fetch_historical_data(asset['symbol'], period, asset['exchange'])
        if df is None or df.empty or len(df) < 200:
            print(f"❌ Insufficient data for {asset['symbol']}")
            return None
        
        # Set date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months_back * 30)
        
        # Filter data
        mask = (df.index >= start_date) & (df.index <= end_date)
        df_period = df[mask].copy()
        
        if len(df_period) < 50:
            print(f"❌ Insufficient data in period for {asset['symbol']}")
            return None
        
        print(f"Data points: {len(df_period)}, Date range: {df_period.index[0]} to {df_period.index[-1]}")
        
        # Calculate indicators for each day
        scores = []
        for i in range(200, len(df_period)):
            window = df_period.iloc[:i+1]
            indicators = TechnicalIndicators.calculate_all_indicators(window)
            
            current_price = float(window.iloc[-1]['Close'])
            score, breakdown, factors = ScoringEngine.calculate_composite_score(
                indicators, current_price, USD_INR_RATE
            )
            
            scores.append({
                'date': window.index[-1],
                'score': score,
                'price': current_price,
                'zone': ScoringEngine.get_zone(score)
            })
        
        scores_df = pd.DataFrame(scores).set_index('date')
        print(f"Calculated scores for {len(scores_df)} days")
        
        # Strategy 1: Regular DCA (baseline)
        regular_dca_config = BacktestConfig(
            symbol=asset['symbol'],
            start_date=start_date,
            end_date=end_date,
            dca_amount=DCA_AMOUNT,
            dca_cadence='monthly',
            buy_dip_threshold=None
        )
        
        regular_result = BacktestEngine.run_backtest(regular_dca_config, df_period)
        
        # Strategy 2: Score-weighted DCA (buy more on dips)
        dip_dca_config = BacktestConfig(
            symbol=asset['symbol'],
            start_date=start_date,
            end_date=end_date,
            dca_amount=DCA_AMOUNT,
            dca_cadence='monthly',
            buy_dip_threshold=60
        )
        
        dip_result = BacktestEngine.run_backtest(dip_dca_config, df_period, scores_df)
        
        # Calculate additional metrics
        print(f"\n📊 RESULTS:")
        print(f"\nRegular DCA (Baseline):")
        print(f"  Total Invested: ₹{regular_result.total_invested:,.2f}")
        print(f"  Final Value: ₹{regular_result.final_value_inr:,.2f}")
        print(f"  Total Return: {regular_result.total_return_pct:.2f}%")
        print(f"  Annualized Return: {regular_result.annualized_return_pct:.2f}%")
        print(f"  Purchases: {regular_result.num_regular_dca}")
        
        print(f"\nScore-Weighted DCA (Buy Dips @ 60+ score):")
        print(f"  Total Invested: ₹{dip_result.total_invested:,.2f}")
        print(f"  Final Value: ₹{dip_result.final_value_inr:,.2f}")
        print(f"  Total Return: {dip_result.total_return_pct:.2f}%")
        print(f"  Annualized Return: {dip_result.annualized_return_pct:.2f}%")
        print(f"  Regular Purchases: {dip_result.num_regular_dca}")
        print(f"  Dip Purchases: {dip_result.num_dip_buys}")
        
        improvement = dip_result.total_return_pct - regular_result.total_return_pct
        print(f"\n✅ Score-Weighted Strategy Improvement: {improvement:+.2f}%")
        
        # Analyze score distribution
        score_stats = scores_df['score'].describe()
        print(f"\n📈 Score Statistics:")
        print(f"  Mean: {score_stats['mean']:.1f}")
        print(f"  Median: {score_stats['50%']:.1f}")
        print(f"  Std Dev: {score_stats['std']:.1f}")
        print(f"  Min: {score_stats['min']:.1f}, Max: {score_stats['max']:.1f}")
        
        # Zone distribution
        zone_counts = scores_df['zone'].value_counts()
        print(f"\n🎯 Zone Distribution:")
        for zone, count in zone_counts.items():
            pct = (count / len(scores_df)) * 100
            print(f"  {zone}: {count} days ({pct:.1f}%)")
        
        return {
            'asset': asset['symbol'],
            'period': period,
            'regular_dca': {
                'return_pct': regular_result.total_return_pct,
                'cagr': regular_result.annualized_return_pct,
                'invested': regular_result.total_invested,
                'final_value': regular_result.final_value_inr
            },
            'score_weighted_dca': {
                'return_pct': dip_result.total_return_pct,
                'cagr': dip_result.annualized_return_pct,
                'invested': dip_result.total_invested,
                'final_value': dip_result.final_value_inr,
                'dip_buys': dip_result.num_dip_buys
            },
            'improvement': improvement,
            'score_stats': {
                'mean': float(score_stats['mean']),
                'median': float(score_stats['50%']),
                'std': float(score_stats['std']),
                'min': float(score_stats['min']),
                'max': float(score_stats['max'])
            },
            'zone_distribution': zone_counts.to_dict()
        }
        
    except Exception as e:
        print(f"❌ Error testing {asset['symbol']}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("="*80)
    print("COMPREHENSIVE DCA SCORING STRATEGY BACKTEST")
    print("="*80)
    print(f"Testing {len(ASSETS)} assets across {len(TIME_HORIZONS)} time horizons")
    print(f"DCA Amount: ₹{DCA_AMOUNT:,} per period")
    
    all_results = []
    
    for period, months in TIME_HORIZONS:
        print(f"\n\n{'#'*80}")
        print(f"# TIME HORIZON: {period} ({months} months)")
        print(f"{'#'*80}")
        
        for asset in ASSETS:
            result = backtest_asset(asset, period, months)
            if result:
                all_results.append(result)
    
    # Save results
    output_file = '/app/backtest_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\n{'='*80}")
    print(f"BACKTEST COMPLETE")
    print(f"Results saved to: {output_file}")
    print(f"Total tests: {len(all_results)}")
    print(f"{'='*80}")
    
    # Summary statistics
    if all_results:
        improvements = [r['improvement'] for r in all_results]
        avg_improvement = np.mean(improvements)
        positive_improvements = sum(1 for i in improvements if i > 0)
        
        print(f"\n📊 SUMMARY:")
        print(f"  Average Improvement: {avg_improvement:+.2f}%")
        print(f"  Positive Results: {positive_improvements}/{len(all_results)} ({(positive_improvements/len(all_results)*100):.1f}%)")
        print(f"  Best Improvement: {max(improvements):+.2f}%")
        print(f"  Worst Improvement: {min(improvements):+.2f}%")

if __name__ == '__main__':
    main()
