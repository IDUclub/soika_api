"""
Router module provides api with the api router for service in swagger interface
and collects clear logic for them.
"""
from typing import Annotated
from fastapi import APIRouter, Query, Depends
from loguru import logger
from app.risk_calculation.dto.project_territory_dto import ProjectTerritoryRequest
from app.risk_calculation.logic.spatial_methods import risk_calculator
from app.risk_calculation.logic.constants import (
    TEXTS,
    CONSTANTS,
    OBJECTS,
    bucket_name,
    text_name,
    constants_name,
    objects_name,
)

calculation_router = APIRouter()


@calculation_router.get("/social_risk/")
async def get_social_risk(
    dto: Annotated[ProjectTerritoryRequest, Depends(ProjectTerritoryRequest)]
) -> dict:
    """Function for calculating social risk for the territory
    Args:
        territory_id (int): ID of the territory
        project_id (int): ID of the project
    Returns:
        dict: table with social risk data
    """
    logger.info(
        f"Started request processing with territory_id={dto.territory_id}, project_id={dto.project_id}"
    )
    TEXTS.try_init(bucket_name, text_name)
    response = await risk_calculator.calculate_social_risk(
        dto.territory_id, dto.project_id
    )
    return response


@calculation_router.get("/risk_coverage_areas/")
async def get_social_risk_coverage(
    dto: Annotated[ProjectTerritoryRequest, Depends(ProjectTerritoryRequest)]
) -> dict:
    """Function for getting outrage coverage for the territory
    Args:
        territory_id (int): ID of the territory
        project_id (int): ID of the project
    Returns:
        dict: dict with two geojsons with coverage areas
    """
    logger.info(
        f"Started request processing with territory_id={dto.territory_id}, project_id={dto.project_id}"
    )
    TEXTS.try_init(bucket_name, text_name)
    response = await risk_calculator.calculate_coverage(
        dto.territory_id, dto.project_id
    )
    logger.info("Social risk coverage response generated")
    return response


@calculation_router.get("/collect_texts/")
async def get_texts_for_territory(
    dto: Annotated[ProjectTerritoryRequest, Depends(ProjectTerritoryRequest)]
) -> dict:
    """Function to collect texts for the territory
    Args:
        territory_id (int): ID of the territory
        project_id (int): ID of the project
    Returns:
        dict: dict with dataframe with texts and their attributes
    """
    logger.info(
        f"Started request processing with territory_id={dto.territory_id}, project_id={dto.project_id}"
    )
    TEXTS.try_init(bucket_name, text_name)
    response = await risk_calculator.collect_texts(dto.territory_id, dto.project_id)
    logger.info("Texts for social risk collected")
    return response


@calculation_router.get("/risk_values/")
async def generate_risk_values_table(
    dto: Annotated[ProjectTerritoryRequest, Depends(ProjectTerritoryRequest)]
) -> dict:
    """Function to generate table for values and risk for the territory
    Args:
        territory_id (int): ID of the territory
        project_id (int): ID of the project
    Returns:
        dict: dict with risk-to-values table
    """
    logger.info(
        f"Started request processing with territory_id={dto.territory_id}, project_id={dto.project_id}"
    )
    TEXTS.try_init(bucket_name, text_name)
    CONSTANTS.try_init(bucket_name, constants_name)
    response = await risk_calculator.calculate_values_to_risk_data(
        dto.territory_id, dto.project_id
    )
    logger.info("Risk-values table generated")
    return response


@calculation_router.get("/named_objects/")
async def get_named_objects(
    dto: Annotated[ProjectTerritoryRequest, Depends(ProjectTerritoryRequest)]
) -> dict:
    """Function to collect named objects for the territory
    Args:
        territory_id (int): ID of the territory
        project_id (int): ID of the project
    Returns:
        dict: dict with geojson
    """
    logger.info(
        f"Started request processing with territory_id={dto.territory_id}, project_id={dto.project_id}"
    )
    OBJECTS.try_init(bucket_name, objects_name)
    response = await risk_calculator.collect_named_objects(
        dto.territory_id, dto.project_id
    )
    logger.info("Named objects collected")
    return response
