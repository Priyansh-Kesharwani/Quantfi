import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import time
import json
from pathlib import Path

from utils.timeout import TimeoutError, run_with_timeout

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    
    results: Dict[str, 'ScorerResult'] = field(default_factory=dict)
    
    failures: Dict[str, str] = field(default_factory=dict)
    
    timings: Dict[str, float] = field(default_factory=dict)
    
    meta: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def successful(self) -> List[str]:
        return list(self.results.keys())
    
    @property
    def failed(self) -> List[str]:
        return list(self.failures.keys())
    
    def get_latest_scores(self) -> Dict[str, float]:
        latest = {}
        for symbol, result in self.results.items():
            valid_scores = result.scores[~np.isnan(result.scores)]
            if len(valid_scores) > 0:
                latest[symbol] = valid_scores[-1]
        return latest
    
    def to_summary_df(self) -> pd.DataFrame:
        rows = []
        
        for symbol in self.successful + self.failed:
            row = {
                'symbol': symbol,
                'status': 'success' if symbol in self.results else 'failed',
                'timing_s': self.timings.get(symbol, None),
                'error': self.failures.get(symbol, None)
            }
            
            if symbol in self.results:
                result = self.results[symbol]
                summary = result.summary()
                row.update({
                    'n_valid': summary['n_valid'],
                    'mean_score': summary['mean_score'],
                    'std_score': summary['std_score'],
                    'pct_above_50': summary['pct_above_50']
                })
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def save_summary(self, path: str) -> None:
        summary = {
            "successful": self.successful,
            "failed": self.failed,
            "timings": self.timings,
            "failures": self.failures,
            "latest_scores": self.get_latest_scores(),
            "meta": self.meta
        }
        
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)


class ScoreRunner:
    
    def __init__(
        self,
        config: Optional['ScorerConfig'] = None,
        indicator_config: Optional['IndicatorConfig'] = None,
        transform_timeout: int = 60,
        fetch_timeout: int = 30,
        max_retries: int = 1,
        log_path: Optional[str] = None
    ):
        from .scorer import ScorerConfig
        
        self.config = config or ScorerConfig()
        self.indicator_config = indicator_config
        self.transform_timeout = transform_timeout
        self.fetch_timeout = fetch_timeout
        self.max_retries = max_retries
        self.log_path = log_path
    
    def run_single(
        self,
        fetcher: 'DataFetcher',
        symbol: str,
        period: str = "max"
    ) -> Tuple[Optional['ScorerResult'], Dict[str, Any]]:
        from .scorer import CompositeScorer
        
        start_time = time.time()
        meta = {
            "symbol": symbol,
            "period": period,
            "status": "pending",
            "error": None
        }
        
        try:
            df, fetch_meta = run_with_timeout(
                fetcher.fetch_daily, 
                self.fetch_timeout,
                symbol, period
            )
            
            if df is None or df.empty:
                meta["status"] = "failed"
                meta["error"] = "No data returned"
                return None, meta
            
            meta["fetch_status"] = fetch_meta.get("status")
            meta["n_rows"] = len(df)
            
        except TimeoutError as e:
            meta["status"] = "failed"
            meta["error"] = f"Fetch timeout: {e}"
            return None, meta
        except Exception as e:
            meta["status"] = "failed"
            meta["error"] = f"Fetch error: {e}"
            return None, meta
        
        try:
            scorer = CompositeScorer(self.config, self.indicator_config)
            
            result = run_with_timeout(
                scorer.fit_transform,
                self.transform_timeout,
                df, debug=True
            )
            
            meta["status"] = "success"
            meta["n_valid_scores"] = result.summary()["n_valid"]
            
        except TimeoutError as e:
            meta["status"] = "failed"
            meta["error"] = f"Transform timeout: {e}"
            return None, meta
        except Exception as e:
            meta["status"] = "failed"
            meta["error"] = f"Transform error: {e}"
            logger.exception(f"Error scoring {symbol}")
            return None, meta
        
        meta["elapsed_s"] = round(time.time() - start_time, 2)
        
        return result, meta
    
    def run_batch(
        self,
        fetcher: 'DataFetcher',
        symbols: List[str],
        period: str = "max"
    ) -> BatchResult:
        logger.info(f"Starting batch scoring for {len(symbols)} assets...")
        
        batch = BatchResult(
            meta={
                "started_at": datetime.utcnow().isoformat(),
                "n_symbols": len(symbols),
                "period": period
            }
        )
        
        for i, symbol in enumerate(symbols):
            logger.info(f"[{i+1}/{len(symbols)}] Scoring {symbol}...")
            
            for attempt in range(self.max_retries + 1):
                result, meta = self.run_single(fetcher, symbol, period)
                
                if result is not None:
                    batch.results[symbol] = result
                    batch.timings[symbol] = meta.get("elapsed_s", 0)
                    break
                else:
                    if attempt < self.max_retries:
                        logger.warning(f"Retry {attempt + 1} for {symbol}...")
                        time.sleep(2 ** attempt)                       
                    else:
                        batch.failures[symbol] = meta.get("error", "Unknown error")
                        batch.timings[symbol] = meta.get("elapsed_s", 0)
        
        batch.meta["completed_at"] = datetime.utcnow().isoformat()
        batch.meta["n_successful"] = len(batch.successful)
        batch.meta["n_failed"] = len(batch.failed)
        
        if self.log_path:
            self._save_log(batch)
        
        logger.info(
            f"Batch complete: {len(batch.successful)} succeeded, "
            f"{len(batch.failed)} failed"
        )
        
        return batch
    
    def _save_log(self, batch: BatchResult) -> None:
        try:
            log_dir = Path(self.log_path).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            batch.save_summary(self.log_path)
        except Exception as e:
            logger.warning(f"Failed to save log: {e}")
