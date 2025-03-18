import aiohttp
from loguru import logger
import geopandas as gpd
import requests

from app.common.config import config
from app.common.exceptions.http_exception_wrapper import http_exception

class TownsnetAPI:
    def __init__(self):
        self.url = config.get("Townsnet_API")
    async def get_evaluated_territories(self, project_id:int, token):
        api_url = f"{self.url}/provision/{project_id}/get_project_evaluation"
        logger.info(f"Fetching evaluated territories from API: {api_url}")
        response = requests.get(api_url, headers={'Authorization': f'Bearer {token}'})
        response.raise_for_status()
        geojson_data = response.json()
        logger.info(f"Evaluated territories for project_id {project_id} successfully fetched from API.")
        territories = gpd.GeoDataFrame.from_features(geojson_data["features"])
        return territories
       
townsnet_api = TownsnetAPI()