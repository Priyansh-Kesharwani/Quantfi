from fastapi import APIRouter, Depends

from backend.models import UserSettings
from backend.routes import get_container
from backend.container import Container

router = APIRouter()


@router.get("/settings", response_model=UserSettings)
async def get_settings(container: Container = Depends(get_container)):
    db = container.db
    settings = await db.user_settings.find_one({}, {"_id": 0})
    if not settings:
        settings = UserSettings().model_dump()
        await db.user_settings.insert_one(settings)
    return UserSettings(**settings)


@router.put("/settings", response_model=UserSettings)
async def update_settings(
    settings: UserSettings,
    container: Container = Depends(get_container),
):
    db = container.db
    doc = settings.model_dump()
    await db.user_settings.update_one(
        {}, {"$set": doc}, upsert=True
    )
    return settings
