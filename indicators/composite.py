import numpy as np
from typing import Tuple, Dict, Any, Optional, List, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import logging

logger = logging.getLogger(__name__)

from indicators.committee import agg_committee


@dataclass
class Phase1Config:
    regime_threshold: Optional[float] = None
    
    regime_threshold_type: str = "hard"
    
    S_scale: Optional[float] = None
    
    committee_method: str = "mean"                               
    committee_trim_pct: float = 0.1
    
    g_pers_type: str = "sigmoid_canonical"                     
    g_pers_params: Dict[str, Optional[float]] = field(default_factory=lambda: {
        "k": 10.0,                                                         
        "H_neutral": 0.5,                                  
        "H_favorable": None,                   
        "H_unfavorable": None,                   
    })
    
    min_obs: int = 100
    
    log_intermediates: bool = True
    log_path: str = "logs/phase1_indicator_runs.json"
    
    allow_production_mode: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime_threshold": self.regime_threshold,
            "regime_threshold_type": self.regime_threshold_type,
            "S_scale": self.S_scale,
            "committee_method": self.committee_method,
            "committee_trim_pct": self.committee_trim_pct,
            "g_pers_type": self.g_pers_type,
            "g_pers_params": self.g_pers_params,
            "min_obs": self.min_obs,
            "log_intermediates": self.log_intermediates,
            "log_path": self.log_path,
            "allow_production_mode": self.allow_production_mode
        }


class CompositeResult(NamedTuple):
    score: float                                  
    T_t: float                            
    U_t: float                                     
    V_t: float                                 
    L_t: float                                
    C_t: float                               
    H_t: float                           
    R_t: float                               
    g_pers_H: float                               
    Gate_t: float                       
    Opp_t: float                               
    RawFavor_t: float                           
    meta: Dict[str, Any]                        


def g_pers(
    H_t: float,
    g_type: str = "sigmoid_canonical",
    params: Optional[Dict[str, Optional[float]]] = None
) -> float:
    if np.isnan(H_t):
        return 0.5                        
    
    if params is None:
        params = {}
    
    if g_type == "sigmoid_canonical":
        k = params.get("k") or 10.0                                                 
        z = k * (H_t - 0.5)
        sigmoid_val = 1.0 / (1.0 + np.exp(-z))
        g = sigmoid_val * 2.0
        return float(np.clip(g, 0.0, 1.0))
    
    elif g_type == "sigmoid":
        H_mid = params.get("H_neutral") or 0.5
        k = params.get("k") or 10.0
        z = k * (H_t - H_mid)
        return float(1.0 / (1.0 + np.exp(-z)))
    
    elif g_type == "linear":
        H_min = params.get("H_unfavorable") or 0.3
        H_max = params.get("H_favorable") or 0.7
        if H_max == H_min:
            return 0.5
        g = (H_t - H_min) / (H_max - H_min)
        return float(np.clip(g, 0.0, 1.0))
    
    elif g_type == "threshold":
        threshold = params.get("threshold") or 0.5
        return 1.0 if H_t > threshold else 0.0
    
    else:
        logger.warning(f"Unknown g_pers type '{g_type}', using sigmoid_canonical")
        k = 10.0
        z = k * (H_t - 0.5)
        sigmoid_val = 1.0 / (1.0 + np.exp(-z))
        return float(np.clip(sigmoid_val * 2.0, 0.0, 1.0))


def compute_gate(
    C_t: float,
    L_t: float,
    R_t: float,
    regime_threshold: Optional[float] = None,
    threshold_type: str = "hard"
) -> Tuple[float, Dict[str, Any]]:
    if np.isnan(C_t):
        C_t = 0.5           
    if np.isnan(L_t):
        L_t = 0.5           
    if np.isnan(R_t):
        R_t = 0.5           
    
    if regime_threshold is not None:
        if threshold_type == "hard":
            R_t_thresholded = 1.0 if R_t >= regime_threshold else 0.0
        else:
            k = 10.0                                                
            R_t_thresholded = 1.0 / (1.0 + np.exp(-k * (R_t - regime_threshold)))
    else:
        R_t_thresholded = R_t
    
    Gate_t = C_t * L_t * R_t_thresholded
    
    meta = {
        "C_t": C_t,
        "L_t": L_t,
        "R_t": R_t,
        "regime_threshold": regime_threshold,
        "threshold_type": threshold_type,
        "R_t_thresholded": R_t_thresholded,
        "Gate_t": Gate_t
    }
    
    return float(Gate_t), meta


def compute_opportunity(
    T_t: float,
    U_t: float,
    H_t: float,
    config: Optional[Phase1Config] = None
) -> Tuple[float, Dict[str, Any]]:
    if config is None:
        config = Phase1Config()
    
    if np.isnan(T_t):
        T_t = 0.5
    if np.isnan(U_t):
        U_t = 0.5
    if np.isnan(H_t):
        H_t = 0.5
    
    g_H = g_pers(H_t, config.g_pers_type, config.g_pers_params)
    
    U_weighted = U_t * g_H
    
    scores = [T_t, U_weighted]
    
    if config.committee_method == "mean":
        Opp_t = np.mean(scores)
        committee_meta = {"method": "mean", "inputs": scores}
    else:
        Opp_t, committee_meta = agg_committee(
            scores,
            method=config.committee_method,
            trim_pct=config.committee_trim_pct
        )
    
    meta = {
        "T_t": T_t,
        "U_t": U_t,
        "H_t": H_t,
        "g_pers_H": g_H,
        "U_weighted": U_weighted,
        "formula": "Opp_t = Mean([T_t, U_t × g_pers(H_t)])",
        "committee_inputs": scores,
        "committee_meta": committee_meta,
        "Opp_t": Opp_t
    }
    
    return float(Opp_t), meta


def compute_composite_score(
    T_t: float,
    U_t: float,
    V_t: float,
    L_t: float,
    C_t: float,
    H_t: float,
    R_t: float,
    config: Optional[Phase1Config] = None
) -> CompositeResult:
    if config is None:
        config = Phase1Config()
    
    Gate_t, gate_meta = compute_gate(
        C_t=C_t,
        L_t=L_t,
        R_t=R_t,
        regime_threshold=config.regime_threshold,
        threshold_type=config.regime_threshold_type
    )
    
    Opp_t, opp_meta = compute_opportunity(
        T_t=T_t,
        U_t=U_t,
        H_t=H_t,
        config=config
    )
    g_pers_H = opp_meta["g_pers_H"]
    
    RawFavor_t = Opp_t * Gate_t
    
    S_scale = config.S_scale if config.S_scale is not None else 1.0
    
    transformed = 0.5 + (RawFavor_t - 0.5) * S_scale
    clipped = float(np.clip(transformed, 0.0, 1.0))
    composite_score = 100.0 * clipped
    
    meta = {
        "config": config.to_dict(),
        "gate_meta": gate_meta,
        "opportunity_meta": opp_meta,
        "RawFavor_t": RawFavor_t,
        "S_scale_used": S_scale,
        "transformed_before_clip": transformed,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if config.log_intermediates:
        _log_computation(
            score=composite_score,
            T_t=T_t, U_t=U_t, V_t=V_t, L_t=L_t, C_t=C_t,
            H_t=H_t, R_t=R_t, Gate_t=Gate_t, Opp_t=Opp_t,
            RawFavor_t=RawFavor_t, g_pers_H=g_pers_H,
            meta=meta,
            log_path=config.log_path
        )
    
    return CompositeResult(
        score=composite_score,
        T_t=T_t if not np.isnan(T_t) else 0.5,
        U_t=U_t if not np.isnan(U_t) else 0.5,
        V_t=V_t if not np.isnan(V_t) else 0.5,
        L_t=L_t if not np.isnan(L_t) else 0.5,
        C_t=C_t if not np.isnan(C_t) else 0.5,
        H_t=H_t if not np.isnan(H_t) else 0.5,
        R_t=R_t if not np.isnan(R_t) else 0.5,
        g_pers_H=g_pers_H,
        Gate_t=Gate_t,
        Opp_t=Opp_t,
        RawFavor_t=RawFavor_t,
        meta=meta
    )


def _log_computation(
    score: float,
    T_t: float, U_t: float, V_t: float, L_t: float, C_t: float,
    H_t: float, R_t: float, Gate_t: float, Opp_t: float,
    RawFavor_t: float, g_pers_H: float,
    meta: Dict[str, Any],
    log_path: str
) -> None:
    try:
        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "composite_score": score,
            "components": {
                "T_t": float(T_t) if not np.isnan(T_t) else None,
                "U_t": float(U_t) if not np.isnan(U_t) else None,
                "V_t": float(V_t) if not np.isnan(V_t) else None,
                "L_t": float(L_t) if not np.isnan(L_t) else None,
                "C_t": float(C_t) if not np.isnan(C_t) else None,
                "H_t": float(H_t) if not np.isnan(H_t) else None,
                "R_t": float(R_t) if not np.isnan(R_t) else None
            },
            "intermediates": {
                "g_pers_H": g_pers_H,
                "Gate_t": Gate_t,
                "Opp_t": Opp_t,
                "RawFavor_t": RawFavor_t
            },
            "config": meta.get("config", {})
        }
        
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
            
    except Exception as e:
        logger.warning(f"Failed to log computation: {e}")


class Phase1Composite:
    
    def __init__(self, config: Optional[Phase1Config] = None):
        self.config = config or Phase1Config()
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Phase1Composite":
        import yaml
        
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        config = Phase1Config(
            regime_threshold=config_dict.get("regime_threshold"),
            S_scale=config_dict.get("S_scale"),
            committee_method=config_dict.get("committee_method", "trimmed_mean"),
            committee_trim_pct=config_dict.get("committee_trim_pct", 0.1),
            g_pers_type=config_dict.get("g_pers_type", "linear"),
            g_pers_params=config_dict.get("g_pers_params", {}),
            min_obs=config_dict.get("min_obs", 100),
            log_intermediates=config_dict.get("log_intermediates", True),
            log_path=config_dict.get("log_path", "logs/phase1_indicator_runs.json"),
            allow_production_mode=config_dict.get("allow_production_mode", False)
        )
        
        return cls(config)
    
    def compute(
        self,
        T_t: float, U_t: float, V_t: float, L_t: float, C_t: float,
        H_t: float, R_t: float
    ) -> CompositeResult:
        return compute_composite_score(
            T_t=T_t, U_t=U_t, V_t=V_t, L_t=L_t, C_t=C_t,
            H_t=H_t, R_t=R_t, config=self.config
        )
    
    def compute_neutral(self) -> CompositeResult:
        return self.compute(
            T_t=0.5, U_t=0.5, V_t=0.5, L_t=0.5, C_t=0.5,
            H_t=0.5, R_t=0.5
        )
