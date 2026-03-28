from fastapi import APIRouter, Depends

from backend.core.container import Container
from backend.routes import get_container

router = APIRouter()


@router.get("/dashboard")
async def get_dashboard(container: Container = Depends(get_container)):
    return await container.dashboard_service.get_snapshot()
