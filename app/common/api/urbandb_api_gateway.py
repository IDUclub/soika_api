import asyncio
import aiohttp
from loguru import logger
from shapely.geometry import shape, Point
import geopandas as gpd
import pandas as pd

from iduconfig import Config
from app.common.exceptions.http_exception_wrapper import http_exception
from app.common.api.api_error_handler import APIHandler
from app.dependencies import config

class UrbanDBAPI:
    def __init__(self, config: Config):
        self.config = config
        self.url = config.get("UrbanDB_API")
        self.session = None
        self.handler = None

    async def init(self):
        self.session = aiohttp.ClientSession()
        self.handler = APIHandler()

    async def close(self):
        if self.session:
            await self.handler.close_session(self.session)
            self.session = None
            logger.info("UrbanDBAPI session closed.")

    async def get_territory_by_name(self, territory_name: str):
        api_url = (
            f"{self.url}/v1/all_territories_without_geometry"
            f"?get_all_levels=true&name={territory_name}&cities_only=false&ordering=asc"
        )
        logger.info(f"Fetching territories from API: {api_url}")
        json_data = await self.handler.request("GET", api_url, session=self.session)
        territory_id = json_data[0]["territory_id"]
        found_territory_name = json_data[0]["name"]
        logger.info(f"For territory {territory_name} found territory_id {territory_id} ({found_territory_name}).")
        return territory_id

    async def get_child_territories(self, territory_id: int, page_num: int, page_size: int):
        api_url = (
            f"{self.url}/v1/territories?parent_id={territory_id}"
            f"&get_all_levels=true&cities_only=False&page={page_num}&page_size={page_size}"
        )
        logger.info(f"Fetching child territories from API: {api_url}")
        json_data = await self.handler.request("GET", api_url, session=self.session)
        logger.info(f"Child territories for territory_id {territory_id} successfully fetched from API.")
        territories = []
        for territory in json_data["results"]:
            territories.append([
                territory["name"],
                territory["territory_id"],
                territory["admin_center"],
                territory["is_city"],
                territory["parent"]["id"],
                territory["level"],
                shape(territory["geometry"]),
            ])
        territories = gpd.GeoDataFrame(
            pd.DataFrame(
                territories,
                columns=[
                    "name",
                    "territory_id",
                    "admin_center",
                    "is_city",
                    "parent_id",
                    "level",
                    "geometry",
                ],
            ),
            geometry="geometry",
            crs=4326,
        )
        territories["admin_center"] = territories["admin_center"].apply(
            lambda x: x.get("id") if isinstance(x, dict) else None
        )
        return territories

    async def get_territory(self, territory_id: int):
        api_url = f"{self.url}/v1/territory/{territory_id}"
        logger.info(f"Fetching territory from API: {api_url}")
        json_data = await self.handler.request("GET", api_url, session=self.session)
        logger.info(f"Territory for territory_id {territory_id} successfully fetched from API.")
        territory = [
            json_data["name"],
            json_data["territory_id"],
            json_data["admin_center"],
            json_data["is_city"],
            json_data["parent"]["id"],
            json_data["level"],
            shape(json_data["geometry"]),
        ]
        territory = gpd.GeoDataFrame(
            pd.DataFrame(
                [territory],
                columns=[
                    "name",
                    "territory_id",
                    "admin_center",
                    "is_city",
                    "parent_id",
                    "level",
                    "geometry",
                ],
            ),
            geometry="geometry",
            crs=4326,
        )
        territory["admin_center"] = await asyncio.to_thread(
            lambda: territory["admin_center"].apply(lambda x: x.get("id") if isinstance(x, dict) else None)
        )
        return territory

    async def get_territories(self, territory_id):
        parent_territory = await self.get_territory(territory_id)
        child_territories = await self.get_child_territories(territory_id, 1, 10000)
        territories = pd.concat([parent_territory, child_territories])
        territories = (
            territories.explode().reset_index(drop=True).drop_duplicates(subset="name")
        )
        return territories

    async def get_project_territory_centroid(self, project_id):
        api_url = f"{self.url}/v1/projects/{project_id}/territory"
        logger.info(f"Fetching project territory from API: {api_url}")
        json_data = await self.handler.request("GET", api_url, session=self.session)
        centre_coords = json_data.get("centre_point", {}).get("coordinates", None)
        return Point(centre_coords)

    async def get_context_ids(self, territory_id, project_id):
        api_url = (
            f"{self.url}/v1/projects?is_regional=false&territory_id={territory_id}"
            f"&page=1&page_size=10000"
        )
        logger.info(f"Fetching list of projects for the territory from API: {api_url}")
        json_data = await self.handler.request("GET", api_url, session=self.session)
        logger.info(f"List of projects for the territory {territory_id} successfully fetched from API.")
        results = json_data.get("results", [])
        for project in results:
            if project.get("project_id") == project_id:
                return project.get("properties", {}).get("context", [])
        raise http_exception(
            404,
            f"Context IDs not found for project {project_id}",
            api_url,
            None
        )

    async def get_context_territories(self, territory_id, project_id):
        context_list = await self.get_context_ids(territory_id, project_id)
        logger.info(f"Starting collection of territories for project {project_id}")
        all_context_territories = []
        chunk_size = 15
        for i in range(0, len(context_list), chunk_size):
            chunk = context_list[i:i + chunk_size]
            tasks = [self.get_territory(context_territory) for context_territory in chunk]
            chunk_results = await asyncio.gather(*tasks)
            all_context_territories.extend(chunk_results)
        context_territories = pd.concat(all_context_territories)
        logger.info(f"Context territories for project {project_id} successfully collected")
        return context_territories
    
    async def get_social_groups(self):
        api_url = (
            f"{self.url}/v1/social_groups"
        )
        results = await self.handler.request("GET", api_url, session=self.session)
        social_groups = pd.DataFrame(results)
        return social_groups

    async def get_services_for_groups(self):
        social_groups = await self.get_social_groups()
        tasks = []
        for _, group in social_groups.iterrows():
            group_id = group['soc_group_id']
            api_url = f"{self.url}/v1/social_groups/{group_id}"
            logger.info(f"Prepared fetch task for social group {group_id}: {api_url}")
            coro = self.handler.request("GET", api_url, session=self.session)
            tasks.append((group['name'], coro))
        responses = await asyncio.gather(*(c for _, c in tasks), return_exceptions=True)
        services_list = []
        for (group_name, _), result in zip(tasks, responses):
            if isinstance(result, Exception):
                logger.error(f"Error fetching services for group {group_name}: {result}")
                continue
            df = pd.DataFrame(result.get('service_types', []))
            df['social_group'] = group_name
            services_list.append(df)
            logger.info(f"Fetched {len(df)} services for group {group_name}")
        if services_list:
            services_data = pd.concat(services_list)
            services_data = services_data.set_index('name')
            logger.info("Service data for social groups successfully collected")
            return services_data
        else:
            logger.warning("No service data collected for any social group")
            return pd.DataFrame()


urban_db_api = UrbanDBAPI(config)