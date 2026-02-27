from datetime import datetime, timedelta, timezone
import logging

from fastapi import APIRouter, HTTPException, Depends

from backend.models import PriceData
from backend.routes import get_container
from backend.container import Container

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/prices/{symbol}")
async def get_latest_price(
    symbol: str,
    container: Container = Depends(get_container),
):
    db = container.db
    cfg = container.config
    asset = await db.assets.find_one(
        {"symbol": symbol.upper()}, {"_id": 0}
    )
    exchange = asset.get("exchange") if asset else None
    currency = asset.get("currency", "USD") if asset else "USD"

    cached = await db.price_history.find_one(
        {"symbol": symbol.upper()},
        {"_id": 0},
        sort=[("timestamp", -1)],
    )
    if cached and isinstance(cached.get("timestamp"), str):
        cached_time = datetime.fromisoformat(cached["timestamp"])
        if (
            datetime.now(timezone.utc)
            - cached_time.replace(tzinfo=timezone.utc)
            < timedelta(minutes=cfg.cache_price_minutes)
        ):
            return cached

    price_provider = container.price_adapter
    price_data = price_provider.fetch_latest_price(symbol.upper(), exchange)
    if not price_data:
        raise HTTPException(
            status_code=404, detail="Unable to fetch price data"
        )
    fx = container.fx_adapter
    usd_inr_rate = fx.fetch_usd_inr_rate() or cfg.fx_fallback_usd_inr
    if currency == "INR":
        price_inr = price_data["price"]
        price_usd = price_inr / usd_inr_rate
    else:
        price_usd = price_data["price"]
        price_inr = price_data["price"] * usd_inr_rate

    price_obj = PriceData(
        symbol=symbol.upper(),
        timestamp=price_data["timestamp"],
        price_usd=price_usd,
        price_inr=price_inr,
        usd_inr_rate=usd_inr_rate,
        volume=price_data.get("volume"),
    )
    doc = price_obj.model_dump()
    doc["timestamp"] = doc["timestamp"].isoformat()
    await db.price_history.insert_one(doc)
    return price_obj


@router.get("/prices/{symbol}/history")
async def get_price_history(
    symbol: str,
    period: str = "1y",
    container: Container = Depends(get_container),
):
    price_provider = container.price_adapter
    df = price_provider.fetch_historical_data(symbol.upper(), period)
    if df is None or df.empty:
        raise HTTPException(
            status_code=404, detail="No historical data available"
        )
    history = []
    for date, row in df.iterrows():
        history.append({
            "date": date.isoformat(),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": float(row["Volume"])
            if "Volume" in row
            else None,
        })
    return {"symbol": symbol.upper(), "history": history}
