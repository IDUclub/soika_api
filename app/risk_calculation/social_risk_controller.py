"""
Router module provides api with the api router for service in swagger interface
and collects clear logic for them.
"""

from fastapi import APIRouter
from loguru import logger
from app.risk_calculation.dto.project_territory_dto import ProjectTerritoryRequest
from app.risk_calculation.logic.spatial_methods import DataStructurer, RiskCalculation
router = APIRouter()


@router.get("/")
async def get_content():
    logger.info("Get content")
    return "Get content"

#--------------------------------------
#Модуль вычисления оценки социального риска
@router.post("/social_risk/")
async def get_stats_by_geom(params: ProjectTerritoryRequest) -> dict[str, dict | list]:
    """Main function for getting statistics for the territory
    Args:
        params (ProjectTerritoryRequest): request in json format from user
    Returns:
        dict[str, dict]: table with data for user
    """
    logger.info(f"Started table request processing with params{params.__dict__}")

    #CITY.try_init(bucket_name, object_name) #загрузка данных с файлового сервера
    #source_data_processed = gpd.read_file('source_data_processed.geojson')
    logger.info("Retrieving geometry from provided territory")
    risk_calculator = RiskCalculation()
    df = await risk_calculator.expand_rows_by_columns(df, columns=['services', 'indicators'])
    df = await risk_calculator.calculate_score(df)
    logger.info(f"Geometry retrieved, hierarchy generated. Starting table generation")
    result_dict = await risk_calculator.score_table(df)
    response = {'Таблица оценки социального риска': result_dict}
    logger.info(f"Tables response generated")
    return response

#Тут происходит загрузка геослоя из базы

#--------------------------------------------------
#Модуль вычисления охвата

# source_data_processed = gpd.read_file('text_data_for_risk_evaluation.geojson')
# df_areas = gpd.read_file('territories_for_risk_evaluation.geojson')

# df_areas = df_areas.merge(source_data_processed['best_match'].value_counts().rename('count'), left_on='name', right_index=True, how='left')
# df_areas = df_areas.sort_values('admin_level').drop_duplicates(subset=['name'], keep='first')
# df_areas.dropna(subset='count', inplace=True)
# df_areas = df_areas[['name', 'admin_level', 'geometry', 'count']]
# coverage_areas = df_areas.to_json()