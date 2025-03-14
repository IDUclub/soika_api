import geopandas as gpd
import json

from loguru import logger
from app.risk_calculation.logic.analysis.constants import TEXTS
from app.common.api.urbandb_api_gateway import urban_db_api
from app.risk_calculation.logic.analysis.geo_utils import geo_utils

class TextProcessing:
    @staticmethod
    async def get_texts(territory_gdf: gpd.GeoDataFrame):
        """
        Извлекает тексты, относящиеся к заданной территории.
        Применяется общая функция для обрезки и приведения CRS.
        """
        texts_gdf = TEXTS.gdf.copy()
        texts_gdf = texts_gdf[texts_gdf["type"] != "post"] # TODO: исключение постов стоит потом пересмотреть
        local_texts = geo_utils.clip_and_reproject(texts_gdf, territory_gdf)
        return {"texts": local_texts}

    async def collect_texts(self, territory_id, project_id):
        logger.info(f"Retrieving texts for project {project_id} and its context")
        project_area = await urban_db_api.get_context_territories(territory_id, project_id)
        texts = await TextProcessing.get_texts(project_area)
        if len(texts['texts']) == 0:
            logger.info(f"No texts for this area")
            response = {}
            return response
        texts_df = texts['texts'].copy()
        texts_df = texts_df[['text', 'services', 'indicators']]
        texts_df = texts_df.groupby(
            ['text', 'services'] 
        ).agg({ 
            'indicators': lambda x: ', '.join(set(x))
        }).reset_index()  
        response = {
            'texts': json.loads(texts_df.to_json()),
        }
        return response
    
text_processing = TextProcessing()