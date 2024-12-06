"""
Router module provides api with the api router for service in swagger interface
and collects clear logic for them.
"""
from fastapi import APIRouter
from loguru import logger
from app.risk_calculation.dto.project_territory_dto import ProjectTerritoryRequest
from app.risk_calculation.logic.spatial_methods import risk_calculator
from app.risk_calculation.logic.constants import TEXTS, bucket_name, text_name


router = APIRouter()


@router.post("/social_risk/")
async def get_social_risk(params: ProjectTerritoryRequest) -> dict[str, dict | list | int]:
    """Function for calculating social risk for the territory
    Args:
        params (ProjectTerritoryRequest): request in json format from user
    Returns:
        dict: table with social risk data
    """
    logger.info(f"Started request processing with params{params.__dict__}")
    TEXTS.try_init(bucket_name, text_name)
    response = await risk_calculator.calculate_social_risk(params.selection_zone)
    return response

@router.post("/risk_coverage_areas/")
async def get_social_risk_coverage(params: ProjectTerritoryRequest) -> dict[str, dict | list | int]:
    """Function fo getting outrage coverage for the territory
    Args:
        params (ProjectTerritoryRequest): request in json format from user
    Returns:
        dict: dict with two geojsons with coverage areas
    """
    logger.info(f"Started request processing with params{params.__dict__}")
    TEXTS.try_init(bucket_name, text_name)
    response = await risk_calculator.calculate_coverage(params.selection_zone, params.territory_id) 
    logger.info(f"Social risk coverage response generated")
    return response