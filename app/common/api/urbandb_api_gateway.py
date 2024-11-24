import aiohttp
from loguru import logger
import json
from shapely.geometry import shape
import geopandas as gpd
import pandas as pd

from app.common.config import config
from app.common.exceptions.http_exception_wrapper import http_exception

class UrbanDBAPI:
    def __init__(self):
        self.url = config.get("UrbanDB_API")

    async def get_child_territories(self, territory_id: int, page_num: int, page_size: int):
        """
        Collecting child territories from UrbanDB_API.
        """
        api_url = f"{self.url}/v1/territories?parent_id={territory_id}&get_all_levels=true&cities_only=False&page={page_num}&page_size={page_size}"
        logger.info(f"Fetching child territories from API: {api_url}")

        async with aiohttp.ClientSession() as session:
            async with session.get(api_url) as response:
                if response.status == 200:
                    pickle_data = await response.read()
                    logger.info(f"Child territories for territory_id {territory_id} successfully fetched from API.")
                    json_data = json.loads(pickle_data.decode('utf-8'))
                    territories = []
                    for territory in json_data['results']:
                        territories.append([territory['name'], shape(territory['geometry'])])
                    territories = gpd.GeoDataFrame(pd.DataFrame(territories, columns=['name', 'geometry']), geometry='geometry', crs=4326)    
                    return territories
                else:
                    logger.error(f"Failed to fetch child territories, status code: {response.status}")
                    raise http_exception(404, f"Failed to fetch child territories", response.status)

    async def get_parent_territory(self, territory_id: int):
        """
        Collecting parent territory from UrbanDB_API.
        """
        api_url = f"{self.url}/v1/territory/{territory_id}"
        logger.info(f"Fetching territories from API: {api_url}")

        async with aiohttp.ClientSession() as session:
            async with session.get(api_url) as response:
                if response.status == 200:
                    pickle_data = await response.read()
                    logger.info(f"Territory for territory_id {territory_id} successfully fetched from API.")
                    json_data = json.loads(pickle_data.decode('utf-8'))
                    territory = [json_data['name'], shape(json_data['geometry'])]
                    territory = gpd.GeoDataFrame(pd.DataFrame([territory], columns = ['name', 'geometry']), geometry='geometry', crs=4326) 
                    return territory
                else:
                    logger.error(f"Failed to fetch parent territory, status code: {response.status}")
                    raise http_exception(404, f"Failed to fetch parent territory", response.status)

    async def get_territories(self, territory_id):
        """
        Collecting all territories from UrbanDB_API
        """
        parent_territory = await self.get_parent_territory(territory_id)
        child_territories = await self.get_child_territories(territory_id, 1, 10000)
        territories = pd.concat([parent_territory, child_territories])
        territories = territories.explode().reset_index(drop=True).drop_duplicates(subset='name')
        return territories          

urban_db_api = UrbanDBAPI()