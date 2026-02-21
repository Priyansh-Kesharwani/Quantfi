"""
Tick aggregation into OHLCV bars with VWAP and bid/ask.

Deterministic: same ticks and freq produce the same output.
"""

from __future__ import annotations

import pandas as pd


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common column names to canonical lower-case names."""
    col_map = {c.lower(): c for c in df.columns}
    out = df.copy()
    renames: dict[str, str] = {}
    if "timestamp_utc" in col_map and col_map["timestamp_utc"] != "timestamp":
        renames[col_map["timestamp_utc"]] = "timestamp"
    elif "ts" in col_map and "timestamp" not in [c.lower() for c in out.columns]:
        renames[col_map["ts"]] = "timestamp"
    if renames:
        out = out.rename(columns=renames)
    return out


def build_bars(ticks: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Aggregate ticks into OHLCV bars with timestamp index (UTC) and optional bid/ask/vwap.

    Parameters
    ----------
    ticks : pd.DataFrame
        Must have a timestamp column (e.g. timestamp, timestamp_utc, ts) and
        price/size. Optional: bid, ask. Timestamps are converted to UTC.
    freq : str
        Pandas offset string for bar period (e.g. "1min", "15min", "1h").

    Returns
    -------
    pd.DataFrame
        Index: DatetimeIndex (UTC). Columns: open, high, low, close, volume,
        bid, ask, vwap.
    """
    df = _normalize_columns(ticks)
    col_lower = {c.lower(): c for c in df.columns}

    # Resolve timestamp
    ts_col = None
    for name in ["timestamp", "timestamp_utc", "ts", "time", "datetime"]:
        if name in df.columns:
            ts_col = name
            break
    if ts_col is None:
        if hasattr(df.index, "dtype") and (
            str(getattr(df.index, "dtype", "")) == "datetime64[ns]"
            or str(getattr(df.index, "dtype", "")).startswith("datetime")
        ):
            ts = pd.Series(df.index, index=df.index, name="ts")
        else:
            raise ValueError("ticks must have a timestamp column or datetime index")
    else:
        ts = pd.to_datetime(df[ts_col], utc=True)
        ts = ts.rename("ts" if ts.name != "ts" else ts.name)

    ts = pd.to_datetime(ts, utc=True)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC", ambiguous="infer")

    # Resolve price/size
    price_col = col_lower.get("price") or col_lower.get("close")
    if not price_col:
        for c in df.columns:
            if "price" in c.lower() or "last" in c.lower():
                price_col = c
                break
        else:
            raise ValueError("ticks must have a price-like column")
    size_col = col_lower.get("size") or col_lower.get("volume")
    price = pd.to_numeric(df[price_col], errors="coerce")
    size = (
        pd.to_numeric(df[size_col], errors="coerce").fillna(1.0)
        if size_col and size_col in df.columns
        else pd.Series(1.0, index=df.index)
    )

    work = pd.DataFrame({"ts": ts.values, "price": price.values, "size": size.values})
    work["ts"] = pd.to_datetime(work["ts"], utc=True)
    work["pv"] = work["price"] * work["size"]

    has_bid = col_lower.get("bid") in df.columns
    has_ask = col_lower.get("ask") in df.columns
    if has_bid and has_ask:
        work["bid"] = pd.to_numeric(df[col_lower["bid"]], errors="coerce").values
        work["ask"] = pd.to_numeric(df[col_lower["ask"]], errors="coerce").values
    else:
        work["bid"] = float("nan")
        work["ask"] = float("nan")

    g = work.groupby(pd.Grouper(key="ts", freq=freq))

    open_ = g["price"].first()
    high = g["price"].max()
    low = g["price"].min()
    close = g["price"].last()
    vol = g["size"].sum()
    vwap_num = g["pv"].sum()
    vwap_den = g["size"].sum().replace(0, float("nan"))
    vwap = vwap_num / vwap_den
    bid = g["bid"].last() if has_bid and has_ask else close * float("nan")
    ask = g["ask"].last() if has_bid and has_ask else close * float("nan")

    out = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
            "bid": bid,
            "ask": ask,
            "vwap": vwap,
        },
        index=open_.index,
    )
    out.index.name = None
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC", ambiguous="infer")
    return out
