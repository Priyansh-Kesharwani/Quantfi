import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import time
import json
from pathlib import Path

from .evaluations import (
    score_vs_forward_returns,
    score_vs_forward_volatility,
    dca_cost_comparison,
    drawdown_analysis,
    crisis_regime_analysis
)
from utils.timeout import TimeoutError, run_with_timeout

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    
    forward_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 60])
    
    n_quantiles: int = 10
    
    dca_high_threshold: float = 70.0
    dca_low_threshold: float = 30.0
    dca_frequency: int = 5
    
    drawdown_threshold: float = -0.10
    
    run_backtest_timeout: int = 180
    transform_timeout: int = 60
    
    crisis_periods: Optional[List[Dict]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "forward_windows": self.forward_windows,
            "n_quantiles": self.n_quantiles,
            "dca_high_threshold": self.dca_high_threshold,
            "dca_low_threshold": self.dca_low_threshold,
            "dca_frequency": self.dca_frequency,
            "drawdown_threshold": self.drawdown_threshold,
            "run_backtest_timeout": self.run_backtest_timeout,
            "transform_timeout": self.transform_timeout
        }


class DecileAnalysis(NamedTuple):
    window: int
    decile_stats: List[Dict]
    monotonicity: float


class DCAComparison(NamedTuple):
    strategies: Dict[str, Dict]
    cost_improvement_pct: float


class CrisisOverlay(NamedTuple):
    crisis_stats: Dict[str, Any]
    normal_stats: Dict[str, Any]
    named_crises: Dict[str, Dict]


@dataclass
class BacktestResult:
    
    symbol: str
    
    decile_analysis: Dict[str, DecileAnalysis] = field(default_factory=dict)
    volatility_analysis: Dict[str, Any] = field(default_factory=dict)
    dca_comparison: Optional[DCAComparison] = None
    drawdown_analysis: Dict[str, Any] = field(default_factory=dict)
    crisis_overlay: Optional[CrisisOverlay] = None
    
    n_observations: int = 0
    n_valid_scores: int = 0
    date_range: Optional[Tuple[str, str]] = None
    
    config: Optional[BacktestConfig] = None
    elapsed_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def summary(self) -> Dict[str, Any]:
        summary = {
            "symbol": self.symbol,
            "n_observations": self.n_observations,
            "n_valid_scores": self.n_valid_scores,
            "date_range": self.date_range,
            "elapsed_seconds": self.elapsed_seconds
        }
        
        if self.decile_analysis:
            summary["decile_monotonicity"] = {
                window: da.monotonicity 
                for window, da in self.decile_analysis.items()
            }
        
        if self.dca_comparison:
            summary["dca_cost_improvement_pct"] = self.dca_comparison.cost_improvement_pct
        
        if self.crisis_overlay:
            summary["crisis_vs_normal_score_diff"] = (
                self.crisis_overlay.crisis_stats.get("mean_score", 0) -
                self.crisis_overlay.normal_stats.get("mean_score", 0)
            )
        
        summary["n_warnings"] = len(self.warnings)
        summary["n_errors"] = len(self.errors)
        
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "n_observations": self.n_observations,
            "n_valid_scores": self.n_valid_scores,
            "date_range": self.date_range,
            "decile_analysis": {
                k: {"window": v.window, "monotonicity": v.monotonicity}
                for k, v in self.decile_analysis.items()
            },
            "dca_comparison": self.dca_comparison._asdict() if self.dca_comparison else None,
            "crisis_overlay": self.crisis_overlay._asdict() if self.crisis_overlay else None,
            "elapsed_seconds": self.elapsed_seconds,
            "warnings": self.warnings,
            "errors": self.errors
        }


class DiagnosticBacktester:
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
    
    def run_backtest(
        self,
        symbol: str,
        scores: np.ndarray,
        prices: np.ndarray,
        dates: Optional[np.ndarray] = None
    ) -> BacktestResult:
        start_time = time.time()
        
        result = BacktestResult(
            symbol=symbol,
            n_observations=len(scores),
            n_valid_scores=int(np.sum(~np.isnan(scores))),
            config=self.config
        )
        
        if dates is not None:
            dates = pd.to_datetime(dates)
            result.date_range = (str(dates.min()), str(dates.max()))
        
        logger.info(f"Running backtest for {symbol} ({result.n_observations} obs)...")
        
        try:
            decile_result = run_with_timeout(
                score_vs_forward_returns,
                self.config.transform_timeout,
                scores, prices,
                self.config.forward_windows,
                self.config.n_quantiles
            )
            
            for window in self.config.forward_windows:
                key = f"{window}d"
                if key in decile_result["decile_analysis"]:
                    mono_key = f"{key}_monotonicity"
                    result.decile_analysis[key] = DecileAnalysis(
                        window=window,
                        decile_stats=decile_result["decile_analysis"][key],
                        monotonicity=decile_result["decile_analysis"].get(mono_key, 0.0)
                    )
                    
        except TimeoutError:
            result.warnings.append("Decile analysis timed out")
        except Exception as e:
            result.errors.append(f"Decile analysis error: {e}")
        
        try:
            vol_result = run_with_timeout(
                score_vs_forward_volatility,
                self.config.transform_timeout,
                scores, prices,
                self.config.forward_windows[:3],                       
                5             
            )
            result.volatility_analysis = vol_result
            
        except TimeoutError:
            result.warnings.append("Volatility analysis timed out")
        except Exception as e:
            result.errors.append(f"Volatility analysis error: {e}")
        
        try:
            dca_result = run_with_timeout(
                dca_cost_comparison,
                self.config.transform_timeout,
                scores, prices,
                self.config.dca_high_threshold,
                self.config.dca_low_threshold,
                1000.0,
                self.config.dca_frequency
            )
            
            result.dca_comparison = DCAComparison(
                strategies=dca_result.get("strategies", {}),
                cost_improvement_pct=dca_result.get("cost_improvement_pct", 0.0)
            )
            
        except TimeoutError:
            result.warnings.append("DCA comparison timed out")
        except Exception as e:
            result.errors.append(f"DCA comparison error: {e}")
        
        try:
            dd_result = run_with_timeout(
                drawdown_analysis,
                self.config.transform_timeout,
                scores, prices,
                self.config.drawdown_threshold
            )
            result.drawdown_analysis = dd_result
            
        except TimeoutError:
            result.warnings.append("Drawdown analysis timed out")
        except Exception as e:
            result.errors.append(f"Drawdown analysis error: {e}")
        
        try:
            crisis_result = run_with_timeout(
                crisis_regime_analysis,
                self.config.transform_timeout,
                scores, prices, dates,
                self.config.crisis_periods
            )
            
            result.crisis_overlay = CrisisOverlay(
                crisis_stats=crisis_result.get("regime_analysis", {}).get("crisis", {}),
                normal_stats=crisis_result.get("regime_analysis", {}).get("normal", {}),
                named_crises=crisis_result.get("named_crises", {})
            )
            
        except TimeoutError:
            result.warnings.append("Crisis analysis timed out")
        except Exception as e:
            result.errors.append(f"Crisis analysis error: {e}")
        
        result.elapsed_seconds = round(time.time() - start_time, 2)
        
        logger.info(
            f"Backtest complete: {result.elapsed_seconds}s, "
            f"{len(result.warnings)} warnings, {len(result.errors)} errors"
        )
        
        return result
    
    def run_batch(
        self,
        data: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]
    ) -> Dict[str, BacktestResult]:
        results = {}
        
        for symbol, (scores, prices, dates) in data.items():
            logger.info(f"Backtesting {symbol}...")
            results[symbol] = self.run_backtest(symbol, scores, prices, dates)
        
        return results
    
    def save_results(
        self,
        result: BacktestResult,
        path: str
    ) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
    
    def print_summary(self, result: BacktestResult) -> str:
        lines = [
            f"\n{'='*60}",
            f"BACKTEST RESULTS: {result.symbol}",
            f"{'='*60}",
            f"Observations: {result.n_observations:,}",
            f"Valid Scores: {result.n_valid_scores:,}",
            f"Date Range: {result.date_range[0]} to {result.date_range[1]}" if result.date_range else "",
            "",
            "DECILE ANALYSIS (Score vs Forward Returns):",
            "-" * 40
        ]
        
        for window, da in result.decile_analysis.items():
            mono_str = f"+{da.monotonicity:.2f}" if da.monotonicity >= 0 else f"{da.monotonicity:.2f}"
            lines.append(f"  {window}: Monotonicity = {mono_str}")
        
        if result.dca_comparison:
            lines.extend([
                "",
                "DCA COST COMPARISON:",
                "-" * 40,
                f"  Cost Improvement: {result.dca_comparison.cost_improvement_pct:.2f}%"
            ])
            for name, stats in result.dca_comparison.strategies.items():
                lines.append(f"  {name}: Avg Cost = ${stats.get('avg_cost_basis', 0):.2f}")
        
        if result.crisis_overlay:
            lines.extend([
                "",
                "CRISIS REGIME ANALYSIS:",
                "-" * 40,
                f"  Crisis Mean Score: {result.crisis_overlay.crisis_stats.get('mean_score', 'N/A')}",
                f"  Normal Mean Score: {result.crisis_overlay.normal_stats.get('mean_score', 'N/A')}"
            ])
        
        if result.warnings:
            lines.extend(["", "WARNINGS:", "-" * 40])
            for w in result.warnings:
                lines.append(f"  ⚠️ {w}")
        
        if result.errors:
            lines.extend(["", "ERRORS:", "-" * 40])
            for e in result.errors:
                lines.append(f"  ❌ {e}")
        
        lines.append(f"\nElapsed: {result.elapsed_seconds}s")
        lines.append("=" * 60)
        
        return "\n".join(lines)
