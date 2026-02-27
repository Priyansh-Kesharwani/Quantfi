"""
Phase B — Data Ingestion & Integrity Validation.

Validates incoming DataFrames against the canonical contract:
  - Index: pd.DatetimeIndex (UTC)
  - Columns: open, high, low, close, volume (+ optional bid, ask, vwap)
  - Monotonic timestamps, consistent bar intervals, bounded NaN rates.

Also provides:
  - fetch_and_validate(): Download from yfinance → validate → canonicalize
  - load_phaseB_config(): Load /config/phaseB.yml
  - Structured error logging on failure
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import logging
import yaml

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}
OPTIONAL_COLUMNS = {"bid", "ask", "vwap"}

def validate_dataframe(
    df: pd.DataFrame,
    symbol: str = "UNKNOWN",
    interval: str = "1d",
    max_nan_pct: float = 0.001,
    bar_interval_tol: float = 0.05,
    required_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Validate a DataFrame against the Phase B data contract.

    Parameters
    ----------
    df : pd.DataFrame
        Input data to validate.
    symbol : str
        Ticker symbol for error logging.
    interval : str
        Expected bar interval (e.g. "1d", "1h").
    max_nan_pct : float
        Maximum allowed fraction of NaN values per column (0.001 = 0.1%).
    bar_interval_tol : float
        Tolerance for bar interval stability (fraction, e.g. 0.05 = ±5%).
    required_columns : list or None
        Override required columns; defaults to REQUIRED_COLUMNS.

    Returns
    -------
    dict
        Keys: symbol, interval, is_valid, errors, warnings, stats.
    """
    req_cols = set(required_columns) if required_columns else REQUIRED_COLUMNS
    errors: List[str] = []
    warnings: List[str] = []
    stats: Dict[str, Any] = {}

    col_lower = {c.lower(): c for c in df.columns}
    present = set(col_lower.keys())
    missing = req_cols - present
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")

    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append("Index is not a DatetimeIndex")
    else:
        if df.index.tz is None:
            warnings.append("Index has no timezone; UTC expected")
        elif str(df.index.tz) != "UTC":
            warnings.append(f"Index timezone is {df.index.tz}, expected UTC")

        if not df.index.is_monotonic_increasing:
            errors.append("Index is not monotonically increasing")

        if len(df) > 2:
            deltas = np.diff(df.index.asi8)
            median_delta = np.median(deltas)
            if median_delta > 0:
                deviations = np.abs(deltas - median_delta) / median_delta
                unstable_pct = float((deviations > bar_interval_tol).mean())
                stats["bar_interval_unstable_pct"] = unstable_pct
                if unstable_pct > 0.10:
                    warnings.append(
                        f"Bar interval unstable: {unstable_pct:.1%} of bars "
                        f"deviate >{bar_interval_tol:.0%} from median"
                    )

    nan_report: Dict[str, float] = {}
    for col_name in req_cols.intersection(present):
        actual_col = col_lower[col_name]
        nan_frac = float(df[actual_col].isna().mean())
        nan_report[col_name] = nan_frac
        if nan_frac > max_nan_pct:
            errors.append(
                f"NaN rate in '{col_name}' = {nan_frac:.4%} "
                f"(max allowed: {max_nan_pct:.4%})"
            )
    stats["nan_rates"] = nan_report

    if "close" in present and "open" in present:
        close = df[col_lower["close"]]
        open_ = df[col_lower["open"]]
        if (close <= 0).any():
            errors.append("Negative or zero close prices detected")
        if (open_ <= 0).any():
            errors.append("Negative or zero open prices detected")

    if "high" in present and "low" in present:
        high = df[col_lower["high"]]
        low = df[col_lower["low"]]
        if (high < low).any():
            n_inv = int((high < low).sum())
            warnings.append(f"High < Low for {n_inv} bars")

    if "volume" in present:
        vol = df[col_lower["volume"]]
        if (vol < 0).any():
            errors.append("Negative volume detected")

    stats["n_rows"] = len(df)
    stats["n_columns"] = len(df.columns)
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
        stats["date_start"] = str(df.index[0])
        stats["date_end"] = str(df.index[-1])

    is_valid = len(errors) == 0

    result = {
        "symbol": symbol,
        "interval": interval,
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
    }

    if not is_valid:
        logger.warning(
            f"Validation FAILED for {symbol}: {json.dumps(errors, indent=2)}"
        )
    else:
        logger.info(
            f"Validation PASSED for {symbol}: "
            f"{stats['n_rows']} rows, {len(warnings)} warnings"
        )

    return result

def clean_dataframe(
    df: pd.DataFrame,
    max_nan_pct: float = 0.001,
    interpolate_method: str = "linear",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean a validated DataFrame: interpolate minor NaN gaps, sort index.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input.
    max_nan_pct : float
        Columns with NaN > this fraction are flagged as unusable.
    interpolate_method : str
        Interpolation method for small gaps.

    Returns
    -------
    cleaned : pd.DataFrame
    report : dict
        Cleaning operations performed.
    """
    report: Dict[str, Any] = {"operations": [], "dropped_rows": 0}

    cleaned = df.copy()

    cleaned.columns = [c.lower() for c in cleaned.columns]
    report["operations"].append("lowered column names")

    if not cleaned.index.is_monotonic_increasing:
        cleaned = cleaned.sort_index()
        report["operations"].append("sorted index")

    n_dupes = int(cleaned.index.duplicated().sum())
    if n_dupes > 0:
        cleaned = cleaned[~cleaned.index.duplicated(keep="first")]
        report["operations"].append(f"removed {n_dupes} duplicate timestamps")

    for col in cleaned.columns:
        nan_frac = float(cleaned[col].isna().mean())
        if 0 < nan_frac <= max_nan_pct:
            cleaned[col] = cleaned[col].interpolate(method=interpolate_method)
            report["operations"].append(
                f"interpolated NaNs in '{col}' ({nan_frac:.4%})"
            )

    if isinstance(cleaned.index, pd.DatetimeIndex):
        if cleaned.index.tz is None:
            cleaned.index = cleaned.index.tz_localize("UTC")
            report["operations"].append("localised index to UTC")
        elif str(cleaned.index.tz) != "UTC":
            old_tz = str(cleaned.index.tz)
            cleaned.index = cleaned.index.tz_convert("UTC")
            report["operations"].append(f"converted index from {old_tz} to UTC")

    return cleaned, report

def canonicalize(
    df: pd.DataFrame,
    symbol: str = "UNKNOWN",
    interval: str = "1d",
    max_nan_pct: float = 0.001,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Validate, clean, and produce a canonical DataFrame.

    Returns
    -------
    canonical_df : pd.DataFrame
        Clean DataFrame with lowercase columns, UTC index.
    report : dict
        Validation + cleaning report.
    """
    validation = validate_dataframe(
        df, symbol=symbol, interval=interval, max_nan_pct=max_nan_pct
    )

    if not validation["is_valid"]:
        cleaned, clean_report = clean_dataframe(df, max_nan_pct=max_nan_pct)
        revalidation = validate_dataframe(
            cleaned, symbol=symbol, interval=interval, max_nan_pct=max_nan_pct
        )
        report = {
            "initial_validation": validation,
            "cleaning": clean_report,
            "post_clean_validation": revalidation,
        }
        return cleaned, report
    else:
        cleaned, clean_report = clean_dataframe(df, max_nan_pct=max_nan_pct)
        report = {
            "initial_validation": validation,
            "cleaning": clean_report,
        }
        return cleaned, report

def fetch_and_validate(
    symbol: str,
    raw_data_dir: str = "raw_data",
    min_history_years: float = 10.0,
    max_nan_pct: float = 0.001,
    save_csv: bool = True,
    period: str = "max",
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """Fetch data from yfinance, validate, canonicalize, and optionally save.

    Bridges the existing ``data/fetcher.py`` (or direct yfinance) with the
    Phase B validation pipeline.

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g. "SPY", "AAPL").
    raw_data_dir : str
        Directory to store raw CSVs (default "raw_data").
    min_history_years : float
        Minimum acceptable history length in years.
    max_nan_pct : float
        Maximum allowed NaN fraction.
    save_csv : bool
        If True, save the raw download to ``{raw_data_dir}/{symbol}.csv``.
    period : str
        yfinance period string (default "max").

    Returns
    -------
    canonical_df : pd.DataFrame or None
        Cleaned, UTC-indexed DataFrame, or None on failure.
    report : dict
        Full pipeline report including fetch, validation, and cleaning steps.
    """
    report: Dict[str, Any] = {
        "symbol": symbol,
        "fetch": {},
        "validation": {},
        "canonical": False,
    }

    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        raw = ticker.history(period=period)
    except ImportError:
        try:
            from data.fetcher import DataFetcher
            fetcher = DataFetcher()
            raw, fetch_meta = fetcher.fetch_daily(symbol, period=period)
            report["fetch"] = fetch_meta
        except Exception as e:
            logger.error(f"All data sources failed for {symbol}: {e}")
            report["fetch"]["error"] = str(e)
            return None, report
    except Exception as e:
        logger.error(f"yfinance fetch failed for {symbol}: {e}")
        report["fetch"]["error"] = str(e)
        return None, report

    if raw is None or raw.empty:
        report["fetch"]["error"] = "Empty result"
        return None, report

    report["fetch"]["rows"] = len(raw)
    report["fetch"]["date_range"] = f"{raw.index[0]} → {raw.index[-1]}"

    if save_csv:
        raw_dir = Path(raw_data_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        csv_path = raw_dir / f"{symbol}.csv"
        raw.to_csv(csv_path)
        report["fetch"]["csv_path"] = str(csv_path)
        logger.info(f"Saved raw CSV: {csv_path}")

    canonical, canon_report = canonicalize(
        raw, symbol=symbol, interval="1d", max_nan_pct=max_nan_pct
    )
    report["validation"] = canon_report

    if len(canonical) > 1:
        years = (canonical.index[-1] - canonical.index[0]).days / 365.25
        report["history_years"] = round(years, 1)
        if years < min_history_years:
            logger.warning(
                f"{symbol}: Only {years:.1f} years of history "
                f"(minimum {min_history_years})"
            )
            report["warning"] = (
                f"Insufficient history: {years:.1f}y < {min_history_years}y"
            )
    else:
        report["history_years"] = 0.0

    report["canonical"] = True
    return canonical, report

def load_phaseB_config(
    path: str = "config/phaseB.yml",
) -> Dict[str, Any]:
    """Load Phase B YAML configuration.

    Parameters
    ----------
    path : str
        Path to the phaseB.yml file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Phase B config not found: {path}")

    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded Phase B config from {path}")
    return config
