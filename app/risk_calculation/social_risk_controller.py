from typing import Annotated
from fastapi import Depends, APIRouter
from loguru import logger
from fastapi.responses import JSONResponse
from app.risk_calculation.dto.project_territory_dto import ProjectTerritoryRequest
from app.risk_calculation.dto.scenario_territory_dto import ScenarioTerritoryRequest
from app.risk_calculation.dto.time_series_dto import TimeSeriesRequest
from app.utils import auth
from app.risk_calculation.social_risk import RiskCalculationService
from app.schema.analysis_response import SocialRiskResponse, CoverageResponse, TextsResponse, RiskValuesResponse, ProvisionToRiskResponse, NamedObjectsResponse, EffectsResponse

calculation_router = APIRouter()

@calculation_router.get("/social_risk/", response_model=SocialRiskResponse)
async def get_social_risk(
    dto: Annotated[ProjectTerritoryRequest, Depends(ProjectTerritoryRequest)]
) -> SocialRiskResponse:
    logger.info(f"Controller: Received social risk request with territory_id={dto.territory_id}, project_id={dto.project_id}")
    response = await RiskCalculationService.get_social_risk(dto.territory_id, dto.project_id)
    return response

@calculation_router.get("/risk_coverage_areas/", response_model=CoverageResponse)
async def get_social_risk_coverage(
    dto: Annotated[ProjectTerritoryRequest, Depends(ProjectTerritoryRequest)]
) -> CoverageResponse:
    logger.info(f"Controller: Received risk coverage request with territory_id={dto.territory_id}, project_id={dto.project_id}")
    response = await RiskCalculationService.get_risk_coverage(dto.territory_id, dto.project_id)
    return response

@calculation_router.get("/collect_texts/", response_model=TextsResponse)
async def get_texts_for_territory(
    dto: Annotated[TimeSeriesRequest, Depends(TimeSeriesRequest)]
) -> TextsResponse:
    logger.info(f"Controller: Received texts request with territory_id={dto.territory_id}, project_id={dto.project_id}, time_period={dto.time_period}")
    response = await RiskCalculationService.collect_texts(dto.territory_id, dto.project_id, dto.time_period)
    return response

@calculation_router.get("/risk_values/", response_model=RiskValuesResponse)
async def generate_risk_values(
    dto: Annotated[ProjectTerritoryRequest, Depends(ProjectTerritoryRequest)]
) -> RiskValuesResponse:
    logger.info(f"Controller: Received risk values request with territory_id={dto.territory_id}, project_id={dto.project_id}")
    response = await RiskCalculationService.generate_risk_values(dto.territory_id, dto.project_id)
    return response

@calculation_router.get('/risk_provision', response_model=ProvisionToRiskResponse)
async def generate_risk_provision(
    dto: Annotated[ProjectTerritoryRequest, Depends(ProjectTerritoryRequest)],
    token: str = Depends(auth.verify_token)
) -> ProvisionToRiskResponse:
    logger.info(f"Controller: Received risk provision request with territory_id={dto.territory_id}, project_id={dto.project_id}")
    response = await RiskCalculationService.generate_risk_provision(dto.territory_id, dto.project_id, token)
    return response

@calculation_router.get("/named_objects/", response_model=NamedObjectsResponse)
async def get_named_objects(
    dto: Annotated[ProjectTerritoryRequest, Depends(ProjectTerritoryRequest)]
) -> NamedObjectsResponse:
    logger.info(f"Controller: Received named objects request with territory_id={dto.territory_id}, project_id={dto.project_id}")
    response = await RiskCalculationService.get_named_objects(dto.territory_id, dto.project_id)
    return response

@calculation_router.get('/risk_effects', response_model=EffectsResponse)
async def get_risks_for_effects(
    dto: ScenarioTerritoryRequest = Depends(),
    token: str = Depends(auth.verify_token),
    service: RiskCalculationService = Depends(),
):
    result = await service.get_risk_effects(
        dto.territory_id, dto.project_id, dto.scenario_id, token
    )

    if result is None:
        return JSONResponse(
            status_code=202,
            content={
                "status": "processing",
                "message": (
                    f"Effects for scenario ID {dto.scenario_id} have started in effects API. "
                    "Please, retry in few minutes."
                )
            }
        )

    return result