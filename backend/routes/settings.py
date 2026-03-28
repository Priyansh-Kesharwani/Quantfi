from fastapi import APIRouter, Depends

from backend.models import UserSettings
from backend.core.container import Container
from backend.routes import get_container

router = APIRouter()


@router.get("/settings", response_model=UserSettings)
async def get_settings(container: Container = Depends(get_container)):
    settings = await container.settings_repo.get()
    if not settings:
        default = UserSettings().model_dump()
        await container.settings_repo.save(default)
        return UserSettings(**default)
    return UserSettings(**settings)


@router.put("/settings", response_model=UserSettings)
async def update_settings(
    settings: UserSettings,
    container: Container = Depends(get_container),
):
    await container.settings_repo.save(settings.model_dump())
    return settings
