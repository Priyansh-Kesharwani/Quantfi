import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import logging

logger = logging.getLogger(__name__)

from engine.indicators.committee import agg_committee

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

@dataclass
class PhaseAConfig:
    """Configuration for Phase A Entry/Exit score composition."""

    committee_method: str = "trimmed_mean"
    committee_trim_pct: float = 0.1
    regime_threshold: Optional[float] = None
    regime_threshold_type: str = "hard"
    S_scale: float = 1.0
    g_pers_type: str = "sigmoid_canonical"
    g_pers_params: Dict[str, Optional[float]] = field(default_factory=lambda: {
        "k": 10.0,
        "H_neutral": 0.5,
        "H_favorable": None,
        "H_unfavorable": None,
    })

    gamma_1: float = 0.4
    gamma_2: float = 0.35
    gamma_3: float = 0.25
    S_scale_exit: float = 1.0

    log_intermediates: bool = True
    log_path: str = "logs/phaseA_indicator_runs.json"

    @classmethod
    def from_yaml(cls, path: str) -> "PhaseAConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        entry = raw.get("composite", {}).get("entry", {})
        exit_ = raw.get("composite", {}).get("exit", {})
        log_cfg = raw.get("logging", {})
        return cls(
            committee_method=entry.get("committee_method", "trimmed_mean"),
            committee_trim_pct=entry.get("committee_trim_pct", 0.1),
            regime_threshold=entry.get("regime_threshold"),
            regime_threshold_type=entry.get("regime_threshold_type", "hard"),
            S_scale=entry.get("S_scale", 1.0),
            g_pers_type=entry.get("g_pers_type", "sigmoid_canonical"),
            g_pers_params=entry.get("g_pers_params", {}),
            gamma_1=exit_.get("gamma_1", 0.4),
            gamma_2=exit_.get("gamma_2", 0.35),
            gamma_3=exit_.get("gamma_3", 0.25),
            S_scale_exit=exit_.get("S_scale_exit", 1.0),
            log_intermediates=log_cfg.get("log_intermediates", True),
            log_path=log_cfg.get("log_path", "logs/phaseA_indicator_runs.json"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "committee_method": self.committee_method,
            "committee_trim_pct": self.committee_trim_pct,
            "regime_threshold": self.regime_threshold,
            "regime_threshold_type": self.regime_threshold_type,
            "S_scale": self.S_scale,
            "g_pers_type": self.g_pers_type,
            "g_pers_params": self.g_pers_params,
            "gamma_1": self.gamma_1,
            "gamma_2": self.gamma_2,
            "gamma_3": self.gamma_3,
            "S_scale_exit": self.S_scale_exit,
        }

def _compute_entry_series(
    components: Dict[str, pd.Series],
    config: PhaseAConfig,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Vectorised Entry Score computation.

    Formula (per spec):
        Opp_t = trimmed_mean([ T_t, U_t · g_pers, P_move_t, OFI_t ])
        Gate_t = 𝟙{R_t ≥ r_thresh} · (1 − S_t) · C_t
        RawFavor_t = Opp_t · Gate_t
        Entry = 100 · clip(0.5 + (RawFavor − 0.5) · S_scale, 0, 1)
    """
    idx = None
    for s in components.values():
        if hasattr(s, "index"):
            idx = s.index
            break

    n = len(next(iter(components.values())))

    T_t = components.get("T_t", pd.Series(np.full(n, 0.5), index=idx))
    U_t = components.get("U_t", pd.Series(np.full(n, 0.5), index=idx))
    H_t = components.get("H_t", pd.Series(np.full(n, 0.5), index=idx))
    R_t = components.get("R_t", pd.Series(np.full(n, 0.5), index=idx))
    C_t = components.get("C_t", pd.Series(np.full(n, 0.5), index=idx))
    OFI_t = components.get("OFI_t", pd.Series(np.full(n, 0.5), index=idx))
    P_move_t = components.get("P_move_t", pd.Series(np.full(n, 0.5), index=idx))
    S_t = components.get("S_t", pd.Series(np.full(n, 0.0), index=idx))

    g_H = pd.Series(
        [g_pers(h, config.g_pers_type, config.g_pers_params) for h in H_t],
        index=idx,
    )

    U_weighted = U_t * g_H

    opp_stack = pd.DataFrame({
        "T_t": T_t.values,
        "U_w": U_weighted.values,
        "P_move": P_move_t.values,
        "OFI": OFI_t.values,
    }, index=idx)

    if config.committee_method == "trimmed_mean":
        from scipy.stats import trim_mean
        Opp_t = opp_stack.apply(
            lambda row: trim_mean(row.dropna().values, config.committee_trim_pct),
            axis=1,
        )
    else:
        Opp_t = opp_stack.mean(axis=1)

    if config.regime_threshold is not None:
        if config.regime_threshold_type == "hard":
            R_pass = (R_t >= config.regime_threshold).astype(float)
        else:
            k_soft = 10.0
            R_pass = 1.0 / (1.0 + np.exp(-k_soft * (R_t - config.regime_threshold)))
    else:
        R_pass = R_t

    Gate_t = R_pass * (1.0 - S_t) * C_t

    RawFavor = Opp_t * Gate_t

    transformed = 0.5 + (RawFavor - 0.5) * config.S_scale
    Entry = 100.0 * transformed.clip(0.0, 1.0)

    breakdown = pd.DataFrame({
        "T_t": T_t.values,
        "U_t": U_t.values,
        "g_H": g_H.values,
        "U_weighted": U_weighted.values,
        "OFI_t": OFI_t.values,
        "P_move_t": P_move_t.values,
        "Opp_t": Opp_t.values,
        "R_t": R_t.values,
        "R_pass": R_pass if isinstance(R_pass, pd.Series) else pd.Series(R_pass, index=idx),
        "C_t": C_t.values,
        "S_t": S_t.values,
        "Gate_t": Gate_t.values,
        "RawFavor": RawFavor.values,
        "Entry_Score": Entry.values,
    }, index=idx)

    return Entry, breakdown

def _compute_exit_series(
    components: Dict[str, pd.Series],
    config: PhaseAConfig,
) -> pd.Series:
    """Vectorised Exit Score computation.

    Formula (per spec):
        Exit_raw = γ₁ · TBL_flag + γ₂ · OFI_rev + γ₃ · λ_decay
        Exit = 100 · clip(0.5 + (Exit_raw − 0.5) · S_scale_exit, 0, 1)
    """
    idx = None
    for s in components.values():
        if hasattr(s, "index"):
            idx = s.index
            break

    n = len(next(iter(components.values())))

    TBL_flag = components.get("TBL_flag", pd.Series(np.full(n, 0.5), index=idx))
    OFI_rev = components.get("OFI_rev", pd.Series(np.full(n, 0.5), index=idx))
    lambda_decay = components.get("lambda_decay", pd.Series(np.full(n, 0.5), index=idx))

    Exit_raw = (
        config.gamma_1 * TBL_flag
        + config.gamma_2 * OFI_rev
        + config.gamma_3 * lambda_decay
    )

    transformed = 0.5 + (Exit_raw - 0.5) * config.S_scale_exit
    Exit = 100.0 * np.clip(transformed, 0.0, 1.0)

    return pd.Series(Exit, index=idx, name="Exit_Score")

def compose_scores(
    components: Dict[str, pd.Series],
    config: Optional[dict] = None,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Phase A composite score engine.

    Computes both Entry_Score and Exit_Score from a dictionary of
    pre-normalised component series.

    Parameters
    ----------
    components : dict
        Keys used by the Entry formula:
            T_t, U_t, H_t, R_t, C_t, OFI_t, P_move_t, S_t
        Keys used by the Exit formula:
            TBL_flag, OFI_rev, lambda_decay
        Missing keys default to 0.5 (neutral).
    config : dict or None
        Raw config dict (e.g. from YAML).  Merged into PhaseAConfig.

    Returns
    -------
    Entry_Score : pd.Series   — values in [0, 100]
    Exit_Score  : pd.Series   — values in [0, 100]
    breakdown_df : pd.DataFrame — intermediate values for audit
    """
    if config is None:
        cfg = PhaseAConfig()
    elif isinstance(config, PhaseAConfig):
        cfg = config
    else:
        entry_cfg = config.get("composite", config).get("entry", config) if isinstance(config.get("composite", None), dict) else config
        exit_cfg = config.get("composite", config).get("exit", config) if isinstance(config.get("composite", None), dict) else config
        cfg = PhaseAConfig(
            committee_method=entry_cfg.get("committee_method", "trimmed_mean"),
            committee_trim_pct=entry_cfg.get("committee_trim_pct", 0.1),
            regime_threshold=entry_cfg.get("regime_threshold"),
            regime_threshold_type=entry_cfg.get("regime_threshold_type", "hard"),
            S_scale=entry_cfg.get("S_scale", 1.0),
            g_pers_type=entry_cfg.get("g_pers_type", "sigmoid_canonical"),
            g_pers_params=entry_cfg.get("g_pers_params", {}),
            gamma_1=exit_cfg.get("gamma_1", 0.4),
            gamma_2=exit_cfg.get("gamma_2", 0.35),
            gamma_3=exit_cfg.get("gamma_3", 0.25),
            S_scale_exit=exit_cfg.get("S_scale_exit", 1.0),
        )

    Entry_Score, breakdown = _compute_entry_series(components, cfg)
    Exit_Score = _compute_exit_series(components, cfg)

    breakdown["Exit_Score"] = Exit_Score.values

    return Entry_Score, Exit_Score, breakdown

def load_phaseA_config(path: str = "config/settings.yml") -> Dict[str, Any]:
    """Load the full Phase A YAML config into a flat dictionary of
    indicator-level and composite-level params.

    Returns a dict with keys::

        seed, normalization.min_obs, normalization.sigmoid_k,
        ofi.window, ofi.normalize, ofi.polarity,
        hawkes.decay, hawkes.dt, hawkes.max_iter, …,
        ldc.kappa, ldc.gamma_default, ldc.feature_window,
        composite  →  PhaseAConfig instance,
        snapshots  →  dict,
        logging    →  dict.

    Usage::

        cfg = load_phaseA_config()
        ofi = compute_ofi(df,
                          window=cfg["ofi"]["window"],
                          normalize=cfg["ofi"]["normalize"],
                          min_obs=cfg["normalization"]["min_obs"])
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    composite_cfg = PhaseAConfig.from_yaml(path)

    return {
        "seed": raw.get("seed", 42),
        "normalization": raw.get("normalization", {}),
        "ofi": raw.get("ofi", {}),
        "hawkes": raw.get("hawkes", {}),
        "ldc": raw.get("ldc", {}),
        "composite": composite_cfg,
        "snapshots": raw.get("snapshots", {}),
        "logging": raw.get("logging", {}),
    }

def load_refactor_config(path=None):
    """Load config/settings.yml."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "config" / "settings.yml"
    with open(path, "r") as f:
        return yaml.safe_load(f)

def g_pers_refactor(H_t: np.ndarray, k_pers: float) -> np.ndarray:
    """g_pers(H_t) = sigmoid(k_pers * (H_t - 0.5)), in (0, 1)."""
    H = np.asarray(H_t, dtype=np.float64)
    z = k_pers * (H - 0.5)
    z = np.clip(z, -500, 500)
    return np.where(np.isnan(H), np.nan, 1.0 / (1.0 + np.exp(-z)))

def _trimmed_mean_row(values: np.ndarray, trim_frac: float) -> float:
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return np.nan
    if trim_frac <= 0:
        return float(np.mean(valid))
    from scipy.stats import trim_mean
    return float(trim_mean(valid, trim_frac))

def compute_opportunity_refactor(
    components,
    trim_frac: float,
    k_pers: float,
    opp_keys=None,
):
    """Opp_t = trimmed_mean(T_t, U_t * g_pers(H_t), LDC_t, O_t)."""
    if opp_keys is None:
        opp_keys = ["T_t", "U_t", "H_t", "LDC_t", "O_t"]
    idx = next(iter(components.values())).index
    n = len(idx)
    get = lambda k: components.get(k, pd.Series(np.full(n, 0.5), index=idx))
    H_t = get("H_t")
    U_t = get("U_t")
    g_H = pd.Series(g_pers_refactor(H_t.values, k_pers), index=idx)
    U_weighted = U_t * g_H

    use_keys = [k for k in opp_keys if k not in ("H_t",)]
    rows = []
    for k in use_keys:
        if k == "U_t":
            rows.append(U_weighted.values)
        else:
            rows.append(get(k).values)
    if not rows:
        return pd.Series(np.full(n, 0.5), index=idx, name="Opp_t")
    mat = np.column_stack(rows)
    opp = np.array([_trimmed_mean_row(mat[i], trim_frac) for i in range(n)])
    return pd.Series(opp, index=idx, name="Opp_t")

def compute_gate_refactor(components, r_thresh: float):
    """Gate_t = C_t * L_t * 1(R_t >= r_thresh)."""
    idx = next(iter(components.values())).index
    n = len(idx)
    get = lambda k: components.get(k, pd.Series(np.full(n, 0.5), index=idx))
    C_t = get("C_t")
    L_t = get("L_t")
    R_t = get("R_t")
    R_pass = (R_t >= r_thresh).astype(float)
    gate = C_t * L_t * R_pass
    return gate.rename("Gate_t")

def compute_composite_score_refactor(components, config=None):
    """Entry CompositeScore and Exit score from refactor config."""
    from engine.indicators.normalization import canonical_normalize

    if config is None:
        config = load_refactor_config()
    comp_cfg = config.get("composite", {})
    exit_cfg = config.get("exit", {})
    trim_frac = comp_cfg.get("trim_frac", 0.1)
    r_thresh = comp_cfg.get("r_thresh", 0.2)
    S_scale = comp_cfg.get("S_scale", 1.0)
    k_pers = comp_cfg.get("k_pers", 6.0)
    gamma1 = exit_cfg.get("gamma1", 1.0)
    gamma2 = exit_cfg.get("gamma2", 1.0)
    gamma3 = exit_cfg.get("gamma3", 1.0)

    idx = next(iter(components.values())).index
    n = len(idx)

    Opp_t = compute_opportunity_refactor(components, trim_frac, k_pers)
    Gate_t = compute_gate_refactor(components, r_thresh)
    RawFavor = Opp_t * Gate_t
    transformed = 0.5 + (RawFavor - 0.5) * S_scale
    Entry_Score = 100.0 * pd.Series(np.clip(transformed.values, 0.0, 1.0), index=idx, name="CompositeScore")

    get = lambda k: components.get(k, pd.Series(np.full(n, 0.5), index=idx))
    TBL_flag = get("TBL_flag")
    OFI_rev = get("OFI_rev")
    lambda_decay = get("lambda_decay")
    ExitRaw = gamma1 * TBL_flag + gamma2 * OFI_rev + gamma3 * lambda_decay
    exit_arr = ExitRaw.values
    exit_norm, _ = canonical_normalize(exit_arr, k=1.0, mode="approx", min_obs=max(1, n // 10))
    Exit_Score = 100.0 * pd.Series(exit_norm, index=idx, name="Exit_Score")

    breakdown = pd.DataFrame({
        "Opp_t": Opp_t.values,
        "Gate_t": Gate_t.values,
        "RawFavor": RawFavor.values,
        "Entry_Score": Entry_Score.values,
        "Exit_Score": Exit_Score.values,
    }, index=idx)

    return Entry_Score, Exit_Score, breakdown

def composite_refactor_from_config(components, config_path=None):
    """Load settings.yml and compute Entry/Exit scores."""
    config = load_refactor_config(config_path)
    return compute_composite_score_refactor(components, config)
