from loguru import logger
import aiohttp
import geopandas as gpd
from iduconfig import Config
from app.common.api.api_error_handler import APIHandler

config = Config()

class TownsnetAPI:
    def __init__(self):
        self.url = config.get("Townsnet_API")
        self.session = aiohttp.ClientSession()
        self.handler = APIHandler(self.session)

    async def get_evaluated_territories(self, project_id: int, token: str):
        api_url = f"{self.url}/provision/{project_id}/get_project_evaluation"
        logger.info(f"Fetching evaluated territories from API: {api_url}")

        headers = {'Authorization': f'Bearer {token}'}
        geojson_data = await self.handler.request("GET", api_url, headers=headers)
        logger.info(
            f"Evaluated territories for project_id {project_id} successfully fetched from API."
        )
        territories = gpd.GeoDataFrame.from_features(geojson_data["features"])
        return territories

townsnet_api = TownsnetAPI()