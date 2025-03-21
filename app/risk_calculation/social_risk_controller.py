from typing import Annotated
from fastapi import Depends, APIRouter
from loguru import logger
from app.risk_calculation.dto.project_territory_dto import ProjectTerritoryRequest
from app.risk_calculation.dto.scenario_territory_dto import ScenarioTerritoryRequest
from app.risk_calculation.dto.time_series_dto import TimeSeriesRequest
from app.utils import auth
from app.risk_calculation.social_risk import RiskCalculationService

calculation_router = APIRouter()

@calculation_router.get("/social_risk/")
async def get_social_risk(
    dto: Annotated[ProjectTerritoryRequest, Depends(ProjectTerritoryRequest)]
) -> dict:
    logger.info(f"Controller: Received social risk request with territory_id={dto.territory_id}, project_id={dto.project_id}")
    response = await RiskCalculationService.get_social_risk(dto.territory_id, dto.project_id)
    return response

@calculation_router.get("/risk_coverage_areas/")
async def get_social_risk_coverage(
    dto: Annotated[ProjectTerritoryRequest, Depends(ProjectTerritoryRequest)]
) -> dict:
    logger.info(f"Controller: Received risk coverage request with territory_id={dto.territory_id}, project_id={dto.project_id}")
    response = await RiskCalculationService.get_risk_coverage(dto.territory_id, dto.project_id)
    return response

@calculation_router.get("/collect_texts/")
async def get_texts_for_territory(
    dto: Annotated[TimeSeriesRequest, Depends(TimeSeriesRequest)]
) -> dict:
    logger.info(f"Controller: Received texts request with territory_id={dto.territory_id}, project_id={dto.project_id}, time_period={dto.time_period}")
    response = await RiskCalculationService.collect_texts(dto.territory_id, dto.project_id, dto.time_period)
    return response

@calculation_router.get("/risk_values/")
async def generate_risk_values(
    dto: Annotated[ProjectTerritoryRequest, Depends(ProjectTerritoryRequest)]
) -> dict:
    logger.info(f"Controller: Received risk values request with territory_id={dto.territory_id}, project_id={dto.project_id}")
    response = await RiskCalculationService.generate_risk_values(dto.territory_id, dto.project_id)
    return response

@calculation_router.get('/risk_provision')
async def generate_risk_provision(
    dto: Annotated[ProjectTerritoryRequest, Depends(ProjectTerritoryRequest)],
    token: str = Depends(auth.verify_token)
) -> dict:
    logger.info(f"Controller: Received risk provision request with territory_id={dto.territory_id}, project_id={dto.project_id}")
    response = await RiskCalculationService.generate_risk_provision(dto.territory_id, dto.project_id, token)
    return response

@calculation_router.get("/named_objects/")
async def get_named_objects(
    dto: Annotated[ProjectTerritoryRequest, Depends(ProjectTerritoryRequest)]
) -> dict:
    logger.info(f"Controller: Received named objects request with territory_id={dto.territory_id}, project_id={dto.project_id}")
    response = await RiskCalculationService.get_named_objects(dto.territory_id, dto.project_id)
    return response

@calculation_router.get('/risk_effects')
async def get_risks_for_effects(
    dto: Annotated[ScenarioTerritoryRequest, Depends(ScenarioTerritoryRequest)],
    token: str = Depends(auth.verify_token)
) -> dict:
    logger.info(f"Controller: Received risk effects request with territory_id={dto.territory_id}, project_id={dto.project_id}, scenario_id={dto.scenario_id}")
    response = await RiskCalculationService.get_risk_effects(dto.territory_id, dto.project_id, dto.scenario_id, token)
    return response