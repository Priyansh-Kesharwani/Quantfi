import logging

from fastapi import APIRouter, HTTPException, Depends

from backend.models import SimulationRequest
from backend.core.container import Container
from backend.routes import get_container

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/simulation/templates")
async def get_simulation_templates(
    container: Container = Depends(get_container),
):
    return {"templates": container.simulation_service.get_templates()}


@router.get("/simulation/cost-presets")
async def get_cost_presets(
    container: Container = Depends(get_container),
):
    return container.simulation_service.get_cost_presets()


@router.post("/simulation/run")
async def run_simulation(
    request: SimulationRequest,
    container: Container = Depends(get_container),
):
    try:
        return await container.simulation_service.run_simulation(
            container.db, request, logger,
        )
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error("Simulation error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
