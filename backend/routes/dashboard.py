from fastapi import APIRouter, Depends

from backend.routes import get_container
from backend.container import Container

router = APIRouter()


@router.get("/dashboard")
async def get_dashboard(container: Container = Depends(get_container)):
    db = container.db
    assets = await db.assets.find(
        {"is_active": True}, {"_id": 0}
    ).to_list(100)
    dashboard_data = []
    for asset in assets:
        symbol = asset["symbol"]
        score = await db.scores.find_one(
            {"symbol": symbol},
            {"_id": 0},
            sort=[("timestamp", -1)],
        )
        price = await db.price_history.find_one(
            {"symbol": symbol},
            {"_id": 0},
            sort=[("timestamp", -1)],
        )
        indicators = await db.indicators.find_one(
            {"symbol": symbol},
            {"_id": 0},
            sort=[("timestamp", -1)],
        )
        dashboard_data.append({
            "asset": asset,
            "score": score,
            "price": price,
            "indicators": indicators,
        })
    return {"assets": dashboard_data}
