import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
import json

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    test_name: str
    passed: bool
    score: float                                     
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'test_name': self.test_name,
            'passed': self.passed,
            'score': self.score,
            'details': self.details,
            'warnings': self.warnings,
            'errors': self.errors
        }


class ScoreValidator:
    
    def __init__(
        self,
        min_coverage: float = 0.8,
        max_score_jump: float = 20.0,                         
        crisis_threshold_low: float = 30.0,
        crisis_threshold_high: float = 70.0
    ):
        self.min_coverage = min_coverage
        self.max_score_jump = max_score_jump
        self.crisis_threshold_low = crisis_threshold_low
        self.crisis_threshold_high = crisis_threshold_high
        
        self.crisis_periods = [
            {'name': 'COVID-19 Crash', 'start': '2020-02-20', 'end': '2020-03-23'},
            {'name': 'GFC 2008', 'start': '2008-09-01', 'end': '2009-03-09'},
            {'name': 'Dot-com Crash', 'start': '2000-03-10', 'end': '2002-10-09'},
            {'name': '2022 Bear Market', 'start': '2022-01-03', 'end': '2022-10-12'}
        ]
    
    def validate_coverage(
        self,
        scores: np.ndarray,
        symbol: str = "unknown"
    ) -> ValidationResult:
        n_total = len(scores)
        n_valid = np.sum(~np.isnan(scores))
        coverage = n_valid / n_total if n_total > 0 else 0
        
        passed = coverage >= self.min_coverage
        
        return ValidationResult(
            test_name="score_coverage",
            passed=passed,
            score=min(100, coverage * 100 / self.min_coverage),
            details={
                'symbol': symbol,
                'total_points': n_total,
                'valid_points': n_valid,
                'coverage': round(coverage * 100, 2),
                'required': self.min_coverage * 100
            },
            warnings=[] if passed else [f"Coverage {coverage:.1%} below {self.min_coverage:.0%} threshold"]
        )
    
    def validate_range(
        self,
        scores: np.ndarray,
        symbol: str = "unknown"
    ) -> ValidationResult:
        valid_scores = scores[~np.isnan(scores)]
        
        if len(valid_scores) == 0:
            return ValidationResult(
                test_name="score_range",
                passed=False,
                score=0,
                details={'symbol': symbol},
                errors=["No valid scores to validate"]
            )
        
        min_score = float(np.min(valid_scores))
        max_score = float(np.max(valid_scores))
        mean_score = float(np.mean(valid_scores))
        
        in_range = (min_score >= 0) and (max_score <= 100)
        out_of_range_count = np.sum((valid_scores < 0) | (valid_scores > 100))
        
        passed = in_range
        
        return ValidationResult(
            test_name="score_range",
            passed=passed,
            score=100 if passed else max(0, 100 - out_of_range_count / len(valid_scores) * 100),
            details={
                'symbol': symbol,
                'min_score': round(min_score, 2),
                'max_score': round(max_score, 2),
                'mean_score': round(mean_score, 2),
                'out_of_range_count': int(out_of_range_count)
            },
            warnings=[] if passed else [f"Scores outside [0, 100]: min={min_score:.2f}, max={max_score:.2f}"]
        )
    
    def validate_smoothness(
        self,
        scores: np.ndarray,
        symbol: str = "unknown"
    ) -> ValidationResult:
        valid_scores = scores[~np.isnan(scores)]
        
        if len(valid_scores) < 2:
            return ValidationResult(
                test_name="score_smoothness",
                passed=False,
                score=0,
                details={'symbol': symbol},
                errors=["Insufficient data for smoothness test"]
            )
        
        score_changes = np.abs(np.diff(valid_scores))
        
        mean_change = float(np.mean(score_changes))
        max_change = float(np.max(score_changes))
        std_change = float(np.std(score_changes))
        
        large_jumps = np.sum(score_changes > self.max_score_jump)
        jump_pct = large_jumps / len(score_changes) * 100
        
        passed = jump_pct < 5                            
        
        return ValidationResult(
            test_name="score_smoothness",
            passed=passed,
            score=max(0, 100 - jump_pct * 10),
            details={
                'symbol': symbol,
                'mean_change': round(mean_change, 2),
                'max_change': round(max_change, 2),
                'std_change': round(std_change, 2),
                'large_jumps_count': int(large_jumps),
                'large_jumps_pct': round(jump_pct, 2)
            },
            warnings=[] if passed else [f"{jump_pct:.1f}% of changes exceed {self.max_score_jump} threshold"]
        )
    
    def validate_crisis_response(
        self,
        scores: np.ndarray,
        dates: np.ndarray,
        symbol: str = "unknown"
    ) -> ValidationResult:
        dates = pd.to_datetime(dates)
        df = pd.DataFrame({'score': scores, 'date': dates}).set_index('date')
        
        crisis_behaviors = []
        total_crisis_days = 0
        extreme_crisis_days = 0
        
        for crisis in self.crisis_periods:
            try:
                start = pd.to_datetime(crisis['start'])
                end = pd.to_datetime(crisis['end'])
                
                mask = (df.index >= start) & (df.index <= end)
                crisis_scores = df.loc[mask, 'score'].dropna()
                
                if len(crisis_scores) > 0:
                    mean_score = float(crisis_scores.mean())
                    min_score = float(crisis_scores.min())
                    max_score = float(crisis_scores.max())
                    
                    extreme_low = np.sum(crisis_scores < self.crisis_threshold_low)
                    extreme_high = np.sum(crisis_scores > self.crisis_threshold_high)
                    
                    total_crisis_days += len(crisis_scores)
                    extreme_crisis_days += extreme_low + extreme_high
                    
                    crisis_behaviors.append({
                        'name': crisis['name'],
                        'days': len(crisis_scores),
                        'mean_score': round(mean_score, 2),
                        'min_score': round(min_score, 2),
                        'max_score': round(max_score, 2),
                        'extreme_days': int(extreme_low + extreme_high)
                    })
            except Exception:
                continue
        
        if total_crisis_days > 0:
            extreme_pct = extreme_crisis_days / total_crisis_days * 100
            passed = extreme_pct > 30                                      
            score = min(100, extreme_pct * 2)
        else:
            passed = True                              
            score = 50                 
            extreme_pct = 0
        
        return ValidationResult(
            test_name="crisis_response",
            passed=passed,
            score=score,
            details={
                'symbol': symbol,
                'crisis_periods_found': len(crisis_behaviors),
                'total_crisis_days': total_crisis_days,
                'extreme_days': extreme_crisis_days,
                'extreme_pct': round(extreme_pct, 2) if total_crisis_days > 0 else None,
                'per_crisis': crisis_behaviors
            },
            warnings=[] if passed else [f"Only {extreme_pct:.1f}% extreme scores during crises"]
        )
    
    def validate_scale_invariance(
        self,
        prices: np.ndarray,
        scorer_func: callable,
        symbol: str = "unknown",
        scale_factor: float = 10.0
    ) -> ValidationResult:
        try:
            original_scores = scorer_func(prices)
            
            scaled_prices = prices * scale_factor
            scaled_scores = scorer_func(scaled_prices)
            
            valid_mask = ~np.isnan(original_scores) & ~np.isnan(scaled_scores)
            
            if np.sum(valid_mask) < 10:
                return ValidationResult(
                    test_name="scale_invariance",
                    passed=False,
                    score=0,
                    details={'symbol': symbol},
                    errors=["Insufficient data for scale invariance test"]
                )
            
            diff = np.abs(original_scores[valid_mask] - scaled_scores[valid_mask])
            mean_diff = float(np.mean(diff))
            max_diff = float(np.max(diff))
            
            passed = mean_diff < 1.0 and max_diff < 5.0
            
            return ValidationResult(
                test_name="scale_invariance",
                passed=passed,
                score=max(0, 100 - mean_diff * 20),
                details={
                    'symbol': symbol,
                    'scale_factor': scale_factor,
                    'mean_difference': round(mean_diff, 3),
                    'max_difference': round(max_diff, 3),
                    'comparison_points': int(np.sum(valid_mask))
                },
                warnings=[] if passed else [f"Score differs by {mean_diff:.2f} avg when prices scaled ×{scale_factor}"]
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="scale_invariance",
                passed=False,
                score=0,
                details={'symbol': symbol},
                errors=[f"Scale invariance test failed: {e}"]
            )
    
    def validate_realtime_signal(
        self,
        scores: np.ndarray,
        prices: np.ndarray,
        symbol: str = "unknown"
    ) -> ValidationResult:
        valid_indices = np.where(~np.isnan(scores))[0]
        
        if len(valid_indices) == 0:
            return ValidationResult(
                test_name="realtime_signal",
                passed=False,
                score=0,
                details={'symbol': symbol},
                errors=["No valid scores available"]
            )
        
        latest_idx = valid_indices[-1]
        latest_score = float(scores[latest_idx])
        
        recency = len(scores) - latest_idx - 1
        is_recent = recency <= 5
        
        is_valid = 0 <= latest_score <= 100
        
        recent_scores = scores[max(0, latest_idx-20):latest_idx+1]
        recent_valid = recent_scores[~np.isnan(recent_scores)]
        
        if len(recent_valid) > 1:
            recent_mean = float(np.mean(recent_valid))
            recent_std = float(np.std(recent_valid))
            is_consistent = abs(latest_score - recent_mean) < 3 * (recent_std + 1)
        else:
            is_consistent = True
            recent_mean = latest_score
            recent_std = 0
        
        if len(prices) > 20:
            recent_prices = prices[max(0, len(prices)-20):]
            recent_vol = np.std(np.diff(np.log(np.maximum(recent_prices, 1e-10)))) * np.sqrt(252)
            vol_context = "high" if recent_vol > 0.25 else ("low" if recent_vol < 0.15 else "normal")
        else:
            recent_vol = None
            vol_context = "unknown"
        
        passed = is_valid and is_recent and is_consistent
        
        return ValidationResult(
            test_name="realtime_signal",
            passed=passed,
            score=100 if passed else 50,
            details={
                'symbol': symbol,
                'latest_score': round(latest_score, 2),
                'recency': int(recency),
                'is_recent': is_recent,
                'is_valid': is_valid,
                'is_consistent': is_consistent,
                'recent_mean': round(recent_mean, 2) if recent_mean else None,
                'recent_std': round(recent_std, 2) if recent_std else None,
                'volatility_regime': vol_context
            },
            warnings=[] if passed else [
                f"Latest score={latest_score:.1f}, recency={recency}, consistent={is_consistent}"
            ]
        )
    
    def validate_interval_consistency(
        self,
        daily_scores: np.ndarray,
        daily_dates: np.ndarray,
        weekly_scores: np.ndarray,
        weekly_dates: np.ndarray,
        symbol: str = "unknown"
    ) -> ValidationResult:
        try:
            daily_df = pd.DataFrame({
                'score': daily_scores,
                'date': pd.to_datetime(daily_dates)
            }).set_index('date')
            
            weekly_df = pd.DataFrame({
                'score': weekly_scores,
                'date': pd.to_datetime(weekly_dates)
            }).set_index('date')
            
            daily_weekly = daily_df['score'].resample('W').mean()
            
            common_dates = daily_weekly.index.intersection(weekly_df.index)
            
            if len(common_dates) < 10:
                return ValidationResult(
                    test_name="interval_consistency",
                    passed=True,                                
                    score=50,
                    details={'symbol': symbol, 'common_weeks': len(common_dates)},
                    warnings=["Insufficient overlapping dates for comparison"]
                )
            
            daily_aligned = daily_weekly.loc[common_dates].dropna()
            weekly_aligned = weekly_df.loc[common_dates, 'score'].dropna()
            
            common_valid = daily_aligned.index.intersection(weekly_aligned.index)
            
            if len(common_valid) < 10:
                return ValidationResult(
                    test_name="interval_consistency",
                    passed=True,
                    score=50,
                    details={'symbol': symbol},
                    warnings=["Insufficient valid data for comparison"]
                )
            
            daily_vals = daily_aligned.loc[common_valid].values
            weekly_vals = weekly_aligned.loc[common_valid].values
            
            correlation = float(np.corrcoef(daily_vals, weekly_vals)[0, 1])
            mean_diff = float(np.mean(np.abs(daily_vals - weekly_vals)))
            
            passed = correlation > 0.7 and mean_diff < 15
            
            return ValidationResult(
                test_name="interval_consistency",
                passed=passed,
                score=max(0, min(100, correlation * 100)),
                details={
                    'symbol': symbol,
                    'correlation': round(correlation, 3),
                    'mean_difference': round(mean_diff, 2),
                    'comparison_weeks': len(common_valid)
                },
                warnings=[] if passed else [
                    f"Interval correlation={correlation:.2f}, mean_diff={mean_diff:.1f}"
                ]
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="interval_consistency",
                passed=False,
                score=0,
                details={'symbol': symbol},
                errors=[f"Interval consistency test failed: {e}"]
            )
    
    def validate_all(
        self,
        scores: np.ndarray,
        prices: np.ndarray,
        dates: np.ndarray,
        symbol: str = "unknown",
        scorer_func: callable = None,
        weekly_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Dict[str, ValidationResult]:
        results = {}
        
        results['coverage'] = self.validate_coverage(scores, symbol)
        
        results['range'] = self.validate_range(scores, symbol)
        
        results['smoothness'] = self.validate_smoothness(scores, symbol)
        
        results['crisis_response'] = self.validate_crisis_response(scores, dates, symbol)
        
        if scorer_func is not None:
            results['scale_invariance'] = self.validate_scale_invariance(
                prices, scorer_func, symbol
            )
        
        results['realtime_signal'] = self.validate_realtime_signal(scores, prices, symbol)
        
        if weekly_data is not None:
            weekly_scores, weekly_dates = weekly_data
            results['interval_consistency'] = self.validate_interval_consistency(
                scores, dates, weekly_scores, weekly_dates, symbol
            )
        
        return results
    
    def summary(self, results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        passed_count = sum(1 for r in results.values() if r.passed)
        total_count = len(results)
        avg_score = np.mean([r.score for r in results.values()])
        
        all_warnings = []
        all_errors = []
        for r in results.values():
            all_warnings.extend(r.warnings)
            all_errors.extend(r.errors)
        
        return {
            'passed': passed_count,
            'total': total_count,
            'pass_rate': f"{passed_count}/{total_count}",
            'avg_score': round(avg_score, 1),
            'all_passed': passed_count == total_count,
            'warnings': all_warnings,
            'errors': all_errors,
            'per_test': {k: v.to_dict() for k, v in results.items()}
        }


def run_all_validations(
    fetch_results: Dict,
    output_path: str = "backtest_logs/validation_results.json"
) -> Dict[str, Any]:
    from score_engine import CompositeScorer, ScorerConfig
    
    validator = ScoreValidator()
    all_results = {}
    
    scorer_config = ScorerConfig(r_thresh=0.5, S_scale=1.5)
    
    symbol_data = {}
    for key, result in fetch_results.items():
        if result.data is None:
            continue
        
        parts = key.rsplit('_', 1)
        symbol = parts[0]
        interval = parts[1] if len(parts) > 1 else '1d'
        
        if symbol not in symbol_data:
            symbol_data[symbol] = {}
        symbol_data[symbol][interval] = result.data
    
    for symbol, intervals in symbol_data.items():
        logger.info(f"Validating {symbol}...")
        
        if '1d' not in intervals:
            logger.warning(f"No daily data for {symbol}, skipping")
            continue
        
        daily_df = intervals['1d']
        
        try:
            scorer = CompositeScorer(scorer_config)
            score_result = scorer.fit_transform(daily_df, debug=False)
            scores = score_result.scores
            prices = daily_df['Close'].values
            dates = daily_df.index.values
            
            weekly_data = None
            if '1wk' in intervals:
                weekly_df = intervals['1wk']
                weekly_scorer = CompositeScorer(scorer_config)
                weekly_result = weekly_scorer.fit_transform(weekly_df, debug=False)
                weekly_data = (weekly_result.scores, weekly_df.index.values)
            
            def scorer_func(prices_arr):
                test_df = pd.DataFrame({
                    'Open': prices_arr,
                    'High': prices_arr * 1.01,
                    'Low': prices_arr * 0.99,
                    'Close': prices_arr,
                    'Volume': np.ones_like(prices_arr) * 1e6
                }, index=daily_df.index[:len(prices_arr)])
                s = CompositeScorer(scorer_config)
                return s.fit_transform(test_df).scores
            
            results = validator.validate_all(
                scores=scores,
                prices=prices,
                dates=dates,
                symbol=symbol,
                scorer_func=scorer_func,
                weekly_data=weekly_data
            )
            
            all_results[symbol] = validator.summary(results)
            
        except Exception as e:
            logger.error(f"Validation failed for {symbol}: {e}")
            all_results[symbol] = {'error': str(e)}
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': all_results
        }, f, indent=2, default=str)
    
    logger.info(f"Validation results saved to {output_path}")
    
    return all_results
