import geopandas as gpd
import json
from loguru import logger
from app.risk_calculation.logic.analysis.constants import OBJECTS
from app.risk_calculation.logic.analysis.geo_utils import geo_utils
from app.common.api.urbandb_api_gateway import urban_db_api

class NamedObjects:        
    @staticmethod
    async def get_objects(territory_gdf: gpd.GeoDataFrame):
        """
        Извлекает объекты для заданной территории с использованием общей функции обрезки.
        """
        objects_gdf = OBJECTS.gdf.copy()
        local_objects = geo_utils.clip_and_reproject(objects_gdf, territory_gdf)
        return {"objects": local_objects.to_crs(epsg=4326)}

    async def collect_named_objects(self, territory_id, project_id):
        logger.info(f"Retrieving objects for project {project_id} and its context")
        project_area = await urban_db_api.get_context_territories(territory_id, project_id)
        objects_result = await NamedObjects.get_objects(project_area)
        local_objects = objects_result['objects']

        if local_objects.empty:
            logger.info("No objects for this area")
            return {}

        response = {
            'named_objects': json.loads(local_objects.to_json())
        }
        return response
    
named_objects_collection = NamedObjects()