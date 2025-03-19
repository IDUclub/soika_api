"""
Router module provides api with the api router for service in swagger interface
and collects clear logic for them.
"""
from typing import Annotated
from fastapi import Depends, APIRouter
from loguru import logger
from app.risk_calculation.dto.project_territory_dto import ProjectTerritoryRequest
from app.risk_calculation.dto.time_series_dto import TimeSeriesRequest
from app.utils import auth
from app.risk_calculation.logic.analysis.social_risk import risk_calculation
from app.risk_calculation.logic.analysis.coverage import coverage_calculation
from app.risk_calculation.logic.analysis.risk_values import risk_values_collection
from app.risk_calculation.logic.analysis.risk_provision import risk_provision_collection
from app.risk_calculation.logic.analysis.texts_processing import text_processing
from app.risk_calculation.logic.analysis.time_series import time_series
from app.risk_calculation.logic.analysis.named_objects import named_objects_collection
from app.risk_calculation.logic.analysis.constants import (
    CONSTANTS,
    bucket_name,
    constants_name
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
    response = await risk_calculation.calculate_social_risk(
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
    response = await coverage_calculation.calculate_coverage(
        dto.territory_id, dto.project_id
    )
    logger.info("Social risk coverage response generated")
    return response


@calculation_router.get("/time_series/")
async def get_time_series(
    dto: Annotated[TimeSeriesRequest, Depends(TimeSeriesRequest)]
) -> dict:
    """Function to collect time series for texts for the territory
    Args:
        territory_id (int): ID of the territory
        project_id (int): ID of the project
        time_period (str): time period to count texts
    Returns:
        dict: dict with dataframe
    """
    logger.info(
        f"Started request processing with territory_id={dto.territory_id}, project_id={dto.project_id}, time_period={dto.time_period}"
    )
    response = await time_series.collect_texts(dto.territory_id, dto.project_id, dto.time_period)
    logger.info("Time series for texts for territory collected")
    return response


@calculation_router.get("/risk_values/")
async def generate_risk_values(
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
    CONSTANTS.try_init(bucket_name, constants_name)
    response = await risk_values_collection.calculate_values_to_risk_data(
        dto.territory_id, dto.project_id
    )
    logger.info("Risk-values table generated")
    return response

@calculation_router.get('/risk_provision')
async def generate_risk_provision(dto: Annotated[ProjectTerritoryRequest, Depends(ProjectTerritoryRequest)],
                        token: str = Depends(auth.verify_token)):
    logger.info(
        f"Started request processing with territory_id={dto.territory_id}, project_id={dto.project_id}"
    )
    response = await risk_provision_collection.calculate_provision_to_risk_data(
        dto.territory_id, dto.project_id, token
    )
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
    response = await named_objects_collection.collect_named_objects(
        dto.territory_id, dto.project_id
    )
    logger.info("Named objects collected")
    return response