import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, NamedTuple
from dataclasses import dataclass, field
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ScorerConfig:
    
    r_thresh: float = 0.5                                            
    
    S_scale: float = 1.5                                
    
    g_pers_k: float = 10.0                     
    
    fill_missing: bool = True
    fill_value: float = 0.5           
    
    enable_geopolitics: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "r_thresh": self.r_thresh,
            "S_scale": self.S_scale,
            "g_pers_k": self.g_pers_k,
            "fill_missing": self.fill_missing,
            "fill_value": self.fill_value,
            "enable_geopolitics": self.enable_geopolitics,
        }


class ScorerResult(NamedTuple):
    
    scores: np.ndarray                             
    
    T_t: np.ndarray
    U_t: np.ndarray
    H_t: np.ndarray
    V_t: np.ndarray
    L_t: np.ndarray
    C_t: np.ndarray
    R_t: np.ndarray
    g_pers_H: np.ndarray
    Opp_t: np.ndarray
    Gate_t: np.ndarray
    RawFavor_t: np.ndarray
    
    index: Optional[pd.DatetimeIndex] = None
    
    meta: Dict[str, Any] = {}
    
    def to_dataframe(self) -> pd.DataFrame:
        data = {
            'CompositeScore': self.scores,
            'T_t': self.T_t,
            'U_t': self.U_t,
            'H_t': self.H_t,
            'V_t': self.V_t,
            'L_t': self.L_t,
            'C_t': self.C_t,
            'R_t': self.R_t,
            'g_pers_H': self.g_pers_H,
            'Opp_t': self.Opp_t,
            'Gate_t': self.Gate_t,
            'RawFavor_t': self.RawFavor_t
        }
        return pd.DataFrame(data, index=self.index)
    
    def get_valid_mask(self) -> np.ndarray:
        return ~np.isnan(self.scores)
    
    def summary(self) -> Dict[str, Any]:
        valid = self.scores[~np.isnan(self.scores)]
        return {
            "n_total": len(self.scores),
            "n_valid": len(valid),
            "mean_score": float(np.mean(valid)) if len(valid) > 0 else None,
            "std_score": float(np.std(valid)) if len(valid) > 0 else None,
            "min_score": float(np.min(valid)) if len(valid) > 0 else None,
            "max_score": float(np.max(valid)) if len(valid) > 0 else None,
            "pct_above_50": float(np.sum(valid > 50) / len(valid) * 100) if len(valid) > 0 else None
        }


class CompositeScorer:
    
    def __init__(
        self,
        config: Optional[ScorerConfig] = None,
        indicator_config: Optional['IndicatorConfig'] = None
    ):
        self.config = config or ScorerConfig()
        self.indicator_config = indicator_config
        
        self._fitted = False
        self._indicators = None
        self._df = None
        self._market_data = None
        self._peer_data = None
    
    def fit(
        self,
        df: pd.DataFrame,
        market_data: Optional[pd.DataFrame] = None,
        peer_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> 'CompositeScorer':
        from indicators.indicator_engine import IndicatorEngine
        
        logger.info(f"Fitting scorer with {len(df)} observations...")
        
        self._df = df
        self._market_data = market_data
        self._peer_data = peer_data
        
        engine = IndicatorEngine(self.indicator_config)
        self._indicators = engine.compute(df, market_data, peer_data)
        
        self._fitted = True
        logger.info("Scorer fitted successfully.")
        
        return self
    
    def transform(self, debug: bool = False) -> ScorerResult:
        if not self._fitted:
            raise RuntimeError("Scorer not fitted. Call .fit() first.")
        
        ind = self._indicators
        n = len(ind.T_t)
        
        logger.info(f"Computing composite scores for {n} observations...")
        
        T_t = self._fill_if_needed(ind.T_t)
        U_t = self._fill_if_needed(ind.U_t)
        H_t = self._fill_if_needed(ind.H_t)
        V_t = self._fill_if_needed(ind.V_t)
        L_t = self._fill_if_needed(ind.L_t)
        C_t = self._fill_if_needed(ind.C_t)
        R_t = self._fill_if_needed(ind.R_t)
        
        k = self.config.g_pers_k
        z = k * (H_t - 0.5)
        g_pers_H = 2.0 / (1.0 + np.exp(-z))
        g_pers_H = np.clip(g_pers_H, 0.0, 1.0)
        
        U_weighted = U_t * g_pers_H
        Opp_t = (T_t + U_weighted) / 2.0
        
        R_threshold = np.where(R_t >= self.config.r_thresh, 1.0, 0.0)
        Gate_t = C_t * L_t * R_threshold
        
        RawFavor_t = Opp_t * Gate_t
        
        G_t = np.ones(n)                    
        if self.config.enable_geopolitics and self._df is not None:
            try:
                from indicators.geopolitics import GeopoliticsEngine
                geo_engine = GeopoliticsEngine()
                if ind.index is not None:
                    G_t = geo_engine.compute_G_t_series(
                        dates=ind.index,
                        symbol=getattr(self, '_symbol', 'GLOBAL'),
                    )
                logger.info(f"G_t applied: mean={np.mean(G_t):.3f}, range=[{np.min(G_t):.3f}, {np.max(G_t):.3f}]")
            except Exception as e:
                logger.warning(f"G_t computation failed, using neutral: {e}")
                G_t = np.ones(n)
        
        S_effective = self.config.S_scale * G_t
        transformed = 0.5 + (RawFavor_t - 0.5) * S_effective
        clipped = np.clip(transformed, 0.0, 1.0)
        scores = 100.0 * clipped
        
        meta = {
            "config": self.config.to_dict(),
            "n_observations": n,
            "n_valid": int(np.sum(~np.isnan(scores))),
            "computed_at": datetime.utcnow().isoformat(),
            "indicator_meta": ind.meta if debug else {},
            "geopolitics_enabled": self.config.enable_geopolitics,
            "G_t_mean": float(np.mean(G_t)),
        }
        
        result = ScorerResult(
            scores=scores,
            T_t=T_t,
            U_t=U_t,
            H_t=H_t,
            V_t=V_t,
            L_t=L_t,
            C_t=C_t,
            R_t=R_t,
            g_pers_H=g_pers_H,
            Opp_t=Opp_t,
            Gate_t=Gate_t,
            RawFavor_t=RawFavor_t,
            index=ind.index,
            meta=meta
        )
        
        logger.info(f"Computed {result.summary()['n_valid']} valid scores.")
        
        return result
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        market_data: Optional[pd.DataFrame] = None,
        peer_data: Optional[Dict[str, pd.DataFrame]] = None,
        debug: bool = False
    ) -> ScorerResult:
        self.fit(df, market_data, peer_data)
        return self.transform(debug=debug)
    
    def _fill_if_needed(self, arr: np.ndarray) -> np.ndarray:
        if self.config.fill_missing:
            return np.where(np.isnan(arr), self.config.fill_value, arr)
        return arr
    
    def score_neutral(self) -> float:
        H_t = 0.5
        k = self.config.g_pers_k
        g_pers = 2.0 / (1.0 + np.exp(-k * (H_t - 0.5)))
        g_pers = min(1.0, g_pers)
        
        T_t, U_t = 0.5, 0.5
        U_weighted = U_t * g_pers
        Opp = (T_t + U_weighted) / 2.0
        
        C_t, L_t, R_t = 0.5, 0.5, 0.5
        R_thresh = 1.0 if R_t >= self.config.r_thresh else 0.0
        Gate = C_t * L_t * R_thresh
        
        Raw = Opp * Gate
        transformed = 0.5 + (Raw - 0.5) * self.config.S_scale
        score = 100.0 * max(0.0, min(1.0, transformed))
        
        return score


def compute_composite_scores(
    df: pd.DataFrame,
    config: Optional[ScorerConfig] = None,
    **kwargs
) -> ScorerResult:
    scorer = CompositeScorer(config)
    return scorer.fit_transform(df, **kwargs)
