"""
Router module provides api with the api router for service in swagger interface
and collects clear logic for them.
"""
import json

from fastapi import APIRouter
from loguru import logger
from app.risk_calculation.dto.project_territory_dto import ProjectTerritoryRequest
from app.risk_calculation.logic.spatial_methods import risk_calculator
from app.risk_calculation.logic.constants import TEXTS, bucket_name, text_name
from app.common.api.urbandb_api_gateway import urban_db_api

router = APIRouter()


@router.post("/social_risk/")
async def get_social_risk(params: ProjectTerritoryRequest) -> dict[str, dict | list]:
    """Function for calculating social risk for the territory
    Args:
        params (ProjectTerritoryRequest): request in json format from user
    Returns:
        dict: table with social risk data
    """
    logger.info(f"Started request processing with params{params.__dict__}")
    TEXTS.try_init(bucket_name, text_name)
    logger.info("Retrieving texts for provided project territory")
    project_area = await risk_calculator.to_gdf(params.selection_zone)
    texts = await risk_calculator.get_texts(project_area)
    if len(texts) == 0:
        logger.info(f"No texts for this area")
        response = {}
        return response
    logger.info("Calculating social risk for provided project territory")
    texts = await risk_calculator.calculate_score(texts)
    result_dict = await risk_calculator.score_table(texts)
    response = {'social_risk_table': result_dict}
    logger.info(f"Table response generated")
    return response

@router.post("/risk_coverage_areas/")
async def get_social_risk_coverage(params: ProjectTerritoryRequest) -> dict[str, dict | list]:
    """Function fo getting outrage coverage for the territory
    Args:
        params (ProjectTerritoryRequest): request in json format from user
    Returns:
        dict: dict with two geojsons with coverage areas
    """
    logger.info(f"Started request processing with params{params.__dict__}")
    TEXTS.try_init(bucket_name, text_name)
    logger.info("Retrieving texts for provided project territory")
    project_area = await risk_calculator.to_gdf(params.selection_zone)
    texts = await risk_calculator.get_texts(project_area)
    if len(texts) == 0:
        logger.info(f"No texts for this area")
        response = {}
        return response
    logger.info("Retrieving potential areas of coverage for provided project territory")
    urban_areas = await urban_db_api.get_territories(params.territory_id)
    logger.info("Calculating coverage")
    urban_areas = await risk_calculator.get_areas(urban_areas, texts)
    logger.info("Generating links from project territory to coverage areas")
    links = await risk_calculator.get_links(project_area, urban_areas)
    response = {
    'coverage_areas': json.loads(urban_areas.to_json()),
    'links_to_project': json.loads(links.to_json())
    }   
    logger.info(f"Social risk coverage response generated")
    return response