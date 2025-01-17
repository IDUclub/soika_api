"""
Router module provides api with the api router for service in swagger interface
and collects clear logic for them.
"""
from fastapi import APIRouter, Query
from loguru import logger
from app.risk_calculation.logic.spatial_methods import risk_calculator
from app.risk_calculation.logic.constants import TEXTS, bucket_name, text_name

router = APIRouter()


@router.get("/social_risk/")
async def get_social_risk(
    territory_id: int = Query(..., description="ID of the territory"),
    project_id: int = Query(..., description="ID of the project")
) -> dict[str, dict | list | int]:
    """Function for calculating social risk for the territory
    Args:
        territory_id (int): ID of the territory
        project_id (int): ID of the project
    Returns:
        dict: table with social risk data
    """
    logger.info(f"Started request processing with territory_id={territory_id}, project_id={project_id}")
    TEXTS.try_init(bucket_name, text_name)
    response = await risk_calculator.calculate_social_risk(territory_id, project_id)
    return response


@router.get("/risk_coverage_areas/")
async def get_social_risk_coverage(
    territory_id: int = Query(..., description="ID of the territory"),
    project_id: int = Query(..., description="ID of the project")
) -> dict[str, dict | list | int]:
    """Function for getting outrage coverage for the territory
    Args:
        territory_id (int): ID of the territory
        project_id (int): ID of the project
    Returns:
        dict: dict with two geojsons with coverage areas
    """
    logger.info(f"Started request processing with territory_id={territory_id}, project_id={project_id}")
    TEXTS.try_init(bucket_name, text_name)
    response = await risk_calculator.calculate_coverage(territory_id, project_id)
    logger.info("Social risk coverage response generated")
    return response


@router.get("/collect_texts/")
async def get_texts_for_territory(
    territory_id: int = Query(..., description="ID of the territory"),
    project_id: int = Query(..., description="ID of the project")
) -> dict[str, dict | list | int]:
    """Function to collect texts for the territory
    Args:
        territory_id (int): ID of the territory
        project_id (int): ID of the project
    Returns:
        dict: dict with dataframe with texts and their attributes
    """
    logger.info(f"Started request processing with territory_id={territory_id}, project_id={project_id}")
    TEXTS.try_init(bucket_name, text_name)
    response = await risk_calculator.collect_texts(territory_id, project_id)
    logger.info("Texts for social risk collected")
    return response
