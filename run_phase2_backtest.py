#!/usr/bin/env python3
"""
Phase 2 Full Backtest Execution Script

Runs full-score computation and diagnostic backtest for all target assets:
- Commodities: XAU, XAG
- US Equities: AAPL, NFLX
- Indian Equities: RELIANCE.NS, TCS.NS

Usage:
    python run_phase2_backtest.py [--sample] [--debug]

Options:
    --sample    Use last 15 years only (faster)
    --debug     Enable debug output

Author: Phase 2 Implementation
Date: 2026-02-07
"""

import sys
import os
import argparse
import logging
import json
from datetime import datetime
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'logs/phase2_backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


# Asset configuration
ASSETS = {
    'commodities': ['XAU', 'XAG'],
    'us_equities': ['AAPL', 'NFLX'],
    'indian_equities': ['RELIANCE.NS', 'TCS.NS']
}


def ensure_directories():
    """Create required directories."""
    dirs = ['logs', 'data/cache', 'results/phase2']
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def run_backtest(sample_mode: bool = False, debug: bool = False):
    """
    Run full backtest for all assets.
    
    Parameters
    ----------
    sample_mode : bool
        If True, use last 15 years only
    debug : bool
        Enable debug output
    """
    from data import DataFetcher
    from score_engine import CompositeScorer, ScorerConfig, ScoreRunner
    from backtester import DiagnosticBacktester, BacktestConfig
    
    start_time = time.time()
    
    # Initialize components
    logger.info("="*70)
    logger.info("PHASE 2 COMPOSITE SCORE BACKTEST")
    logger.info("="*70)
    logger.info(f"Mode: {'SAMPLE (15y)' if sample_mode else 'FULL HISTORY'}")
    logger.info(f"Debug: {debug}")
    
    fetcher = DataFetcher(cache_dir='data/cache')
    
    scorer_config = ScorerConfig(
        r_thresh=0.5,
        S_scale=1.5,
        g_pers_k=10.0,
        fill_missing=True
    )
    
    backtest_config = BacktestConfig(
        forward_windows=[5, 10, 20, 60],
        n_quantiles=10,
        dca_high_threshold=70,
        dca_low_threshold=30,
        run_backtest_timeout=180,
        transform_timeout=60
    )
    
    backtester = DiagnosticBacktester(backtest_config)
    
    # Results storage
    all_results = {
        'run_info': {
            'timestamp': datetime.now().isoformat(),
            'sample_mode': sample_mode,
            'assets': ASSETS
        },
        'scores': {},
        'backtests': {},
        'summary': {}
    }
    
    # Process each asset class
    all_symbols = []
    for asset_class, symbols in ASSETS.items():
        all_symbols.extend(symbols)
    
    logger.info(f"\nProcessing {len(all_symbols)} assets: {all_symbols}")
    
    for idx, symbol in enumerate(all_symbols):
        logger.info(f"\n[{idx+1}/{len(all_symbols)}] Processing {symbol}...")
        
        try:
            # 1. Fetch data
            logger.info(f"  Fetching historical data...")
            period = '15y' if sample_mode else 'max'
            df, meta = fetcher.fetch_daily(symbol, period=period)
            
            if df is None or df.empty:
                logger.warning(f"  No data for {symbol}, skipping")
                continue
            
            n_years = (df.index.max() - df.index.min()).days / 365.25
            logger.info(f"  ✓ {len(df)} observations ({n_years:.1f} years)")
            
            # 2. Compute scores
            logger.info(f"  Computing composite scores...")
            scorer = CompositeScorer(scorer_config)
            score_result = scorer.fit_transform(df, debug=debug)
            
            summary = score_result.summary()
            logger.info(f"  ✓ {summary['n_valid']} valid scores")
            logger.info(f"    Mean: {summary['mean_score']:.1f}, "
                       f"Std: {summary['std_score']:.1f}, "
                       f"Above 50: {summary['pct_above_50']:.1f}%")
            
            # Store score summary
            all_results['scores'][symbol] = {
                'n_observations': summary['n_total'],
                'n_valid': summary['n_valid'],
                'mean_score': summary['mean_score'],
                'std_score': summary['std_score'],
                'pct_above_50': summary['pct_above_50'],
                'date_range': [str(df.index.min()), str(df.index.max())]
            }
            
            # 3. Run diagnostic backtest
            logger.info(f"  Running diagnostic backtest...")
            bt_result = backtester.run_backtest(
                symbol=symbol,
                scores=score_result.scores,
                prices=df['Close'].values,
                dates=df.index.values
            )
            
            logger.info(f"  ✓ Backtest complete ({bt_result.elapsed_seconds:.1f}s)")
            
            # Store backtest summary
            all_results['backtests'][symbol] = {
                'elapsed_seconds': bt_result.elapsed_seconds,
                'decile_monotonicity': {
                    k: v.monotonicity for k, v in bt_result.decile_analysis.items()
                },
                'dca_cost_improvement': (
                    bt_result.dca_comparison.cost_improvement_pct 
                    if bt_result.dca_comparison else None
                ),
                'warnings': bt_result.warnings,
                'errors': bt_result.errors
            }
            
            # Print summary
            print(backtester.print_summary(bt_result))
            
        except Exception as e:
            logger.error(f"  ✗ Error processing {symbol}: {e}")
            all_results['backtests'][symbol] = {'error': str(e)}
    
    # Generate overall summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    
    successful = [s for s in all_symbols if s in all_results['scores']]
    failed = [s for s in all_symbols if s not in all_results['scores']]
    
    all_results['summary'] = {
        'total_assets': len(all_symbols),
        'successful': len(successful),
        'failed': len(failed),
        'failed_symbols': failed,
        'total_runtime_seconds': time.time() - start_time
    }
    
    logger.info(f"Total assets: {len(all_symbols)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    if failed:
        logger.info(f"Failed symbols: {failed}")
    logger.info(f"Total runtime: {all_results['summary']['total_runtime_seconds']:.1f}s")
    
    # Print effectiveness table
    print("\n" + "="*80)
    print("SCORE EFFECTIVENESS TABLE")
    print("="*80)
    print(f"{'Symbol':<15} {'Years':>8} {'Valid':>8} {'Mean':>8} {'Mono-5d':>10} {'Mono-10d':>10} {'DCA Imp':>10}")
    print("-"*80)
    
    for symbol in successful:
        score_info = all_results['scores'].get(symbol, {})
        bt_info = all_results['backtests'].get(symbol, {})
        
        n_years = (
            (pd.to_datetime(score_info.get('date_range', ['', ''])[1]) - 
             pd.to_datetime(score_info.get('date_range', ['', ''])[0])).days / 365.25
            if score_info.get('date_range') else 0
        )
        
        mono = bt_info.get('decile_monotonicity', {})
        dca_imp = bt_info.get('dca_cost_improvement', None)
        
        print(f"{symbol:<15} {n_years:>8.1f} {score_info.get('n_valid', 0):>8} "
              f"{score_info.get('mean_score', 0):>8.1f} "
              f"{mono.get('5d', 0):>10.2f} {mono.get('10d', 0):>10.2f} "
              f"{dca_imp:>10.2f}%" if dca_imp else f"{symbol:<15} {n_years:>8.1f} {score_info.get('n_valid', 0):>8} "
              f"{score_info.get('mean_score', 0):>8.1f} "
              f"{mono.get('5d', 0):>10.2f} {mono.get('10d', 0):>10.2f} {'N/A':>10}")
    
    print("="*80)
    
    # Save results
    results_path = f"results/phase2/backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to: {results_path}")
    
    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Phase 2 Composite Score Backtest'
    )
    parser.add_argument('--sample', action='store_true',
                       help='Use last 15 years only (faster)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    
    args = parser.parse_args()
    
    ensure_directories()
    
    try:
        results = run_backtest(
            sample_mode=args.sample,
            debug=args.debug
        )
        
        # Exit with status based on results
        if results['summary']['failed'] == 0:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\nBacktest interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Backtest failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
