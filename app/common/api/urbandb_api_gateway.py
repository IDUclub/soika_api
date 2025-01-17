import aiohttp
from loguru import logger
import json
from shapely.geometry import shape, Point
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
                    json_data = await response.json()
                    logger.info(f"Child territories for territory_id {territory_id} successfully fetched from API.")
                    territories = []
                    for territory in json_data['results']:
                        territories.append([territory['name'], shape(territory['geometry'])])
                    territories = gpd.GeoDataFrame(pd.DataFrame(territories, columns=['name', 'geometry']), geometry='geometry', crs=4326)    
                    return territories
                else:
                    logger.error(f"Failed to fetch child territories, status code: {response.status}")
                    raise http_exception(404, f"Failed to fetch child territories", response.status)

    async def get_territory(self, territory_id: int):
        """
        Collecting parent territory from UrbanDB_API.
        """
        api_url = f"{self.url}/v1/territory/{territory_id}"
        logger.info(f"Fetching territories from API: {api_url}")

        async with aiohttp.ClientSession() as session:
            async with session.get(api_url) as response:
                if response.status == 200:
                    json_data = await response.json()
                    logger.info(f"Territory for territory_id {territory_id} successfully fetched from API.")
                    territory = [json_data['name'], shape(json_data['geometry'])]
                    territory = gpd.GeoDataFrame(pd.DataFrame([territory], columns = ['name', 'geometry']), geometry='geometry', crs=4326) 
                    return territory
                else:
                    logger.error(f"Failed to fetch territory, status code: {response.status}")
                    raise http_exception(404, f"Failed to fetch territory", response.status)

    async def get_territories(self, territory_id):
        """
        Collecting all territories from UrbanDB_API
        """
        parent_territory = await self.get_territory(territory_id)
        child_territories = await self.get_child_territories(territory_id, 1, 10000)
        territories = pd.concat([parent_territory, child_territories])
        territories = territories.explode().reset_index(drop=True).drop_duplicates(subset='name')
        return territories

    async def get_project_territory_centroid(self, project_id):
        """
        Fetching centroid of territory of the project
        """
        api_url = f"{self.url}/v1/projects/{project_id}/territory"
        logger.info(f"Fetching project territory from API: {api_url}")

        async with aiohttp.ClientSession() as session:
            async with session.get(api_url) as response:
                if response.status == 200:
                    json_data = await response.json()
                    centre_coords = json_data.get("centre_point", {}).get("coordinates", None)
                    return Point(centre_coords)
                else:
                    logger.error(f"Failed to fetch centroid of territory, status code: {response.status}")
                    raise http_exception(404, f"Failed to fetch centroid of territory", response.status)

    async def get_context_ids(self, territory_id, project_id):
        """
        Finding id of terrotories which are in the context of the project
        """
        api_url = f"{self.url}/v1/projects?is_regional=false&territory_id={territory_id}&page=1&page_size=10000"          
        logger.info(f"Fetching list of projects for the territory from API: {api_url}")

        async with aiohttp.ClientSession() as session:
            async with session.get(api_url) as response:
                if response.status == 200:
                    json_data = await response.json()
                    logger.info(f"List of projects for the territory {territory_id} successfully fetched from API.")
                    results = json_data.get("results", [])
                    for project in results:
                        if project.get("project_id") == project_id:
                            return project.get("properties", {}).get("context", [])
                    raise http_exception(404, f"Failed to fetch list of context ids", response.status)
                else:
                    logger.error(f"Failed to fetch list of projects, status code: {response.status}")
                    raise http_exception(404, f"Failed to fetch list of projects", response.status)

    async def get_context_territories(self, territory_id, project_id):
        """
        Collecting parent territory from UrbanDB_API.
        """
        context_list = await self.get_context_ids(territory_id, project_id)
        context_territories = [await self.get_territory(territory_id) for context_territory in context_list]
        context_territories = pd.concat(context_territories)
        logger.info(f"Context territories for project {project_id} succesfully collected")
        return context_territories
                
urban_db_api = UrbanDBAPI()