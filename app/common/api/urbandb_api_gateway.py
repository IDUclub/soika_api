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
from app.utils import auth

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

    async def _auth_headers(
        self,
        actor_user_id: str | None = None,
        user_token: str | None = None,
    ) -> dict[str, str]:
        return await auth.get_outgoing_auth_headers(
            actor_user_id=actor_user_id,
            user_token=user_token,
        )

    async def ensure_project_access(self, territory_id: int, project_id: int, user_token: str):
        await self.get_context_ids(territory_id, project_id, user_token=user_token)

    async def ensure_territory_access(
        self,
        territory_id: int | None,
        territory_name: str | None,
        user_token: str,
    ) -> None:
        if territory_id is not None:
            await self.get_territory(territory_id, user_token=user_token)
            return

        if territory_name is None:
            raise http_exception(
                400,
                "Either territory_id or territory_name is required",
                None,
                None,
            )

        await self.get_territory_by_name(territory_name, user_token=user_token)

    async def get_territory_by_name(
        self,
        territory_name: str,
        actor_user_id: str | None = None,
        user_token: str | None = None,
    ):
        api_url = (
            f"{self.url}/v1/all_territories_without_geometry"
            f"?get_all_levels=true&name={territory_name}&cities_only=false&ordering=asc"
        )
        logger.info(f"Fetching territories from API: {api_url}")
        headers = await self._auth_headers(actor_user_id=actor_user_id, user_token=user_token)
        json_data = await self.handler.request("GET", api_url, session=self.session, headers=headers)
        territory_id = json_data[0]["territory_id"]
        found_territory_name = json_data[0]["name"]
        logger.info(f"For territory {territory_name} found territory_id {territory_id} ({found_territory_name}).")
        return territory_id

    async def get_child_territories(
        self,
        territory_id: int,
        page_num: int,
        page_size: int,
        actor_user_id: str | None = None,
        user_token: str | None = None,
    ):
        api_url = (
            f"{self.url}/v1/territories?parent_id={territory_id}"
            f"&get_all_levels=true&cities_only=False&page={page_num}&page_size={page_size}"
        )
        logger.info(f"Fetching child territories from API: {api_url}")
        headers = await self._auth_headers(actor_user_id=actor_user_id, user_token=user_token)
        json_data = await self.handler.request("GET", api_url, session=self.session, headers=headers)
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

    async def get_territory(
        self,
        territory_id: int,
        actor_user_id: str | None = None,
        user_token: str | None = None,
    ):
        api_url = f"{self.url}/v1/territory/{territory_id}"
        logger.info(f"Fetching territory from API: {api_url}")
        headers = await self._auth_headers(actor_user_id=actor_user_id, user_token=user_token)
        json_data = await self.handler.request("GET", api_url, session=self.session, headers=headers)
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

    async def get_territories(self, territory_id, actor_user_id: str | None = None):
        parent_territory = await self.get_territory(territory_id, actor_user_id=actor_user_id)
        child_territories = await self.get_child_territories(
            territory_id,
            1,
            10000,
            actor_user_id=actor_user_id,
        )
        territories = pd.concat([parent_territory, child_territories])
        territories = (
            territories.explode().reset_index(drop=True).drop_duplicates(subset="name")
        )
        return territories

    async def get_project_territory_centroid(self, project_id, actor_user_id: str | None = None):
        api_url = f"{self.url}/v1/projects/{project_id}/territory"
        logger.info(f"Fetching project territory from API: {api_url}")
        headers = await self._auth_headers(actor_user_id=actor_user_id)
        json_data = await self.handler.request("GET", api_url, session=self.session, headers=headers)
        centre_coords = json_data.get("centre_point", {}).get("coordinates", None)
        return Point(centre_coords)

    async def get_context_ids(
        self,
        territory_id,
        project_id,
        actor_user_id: str | None = None,
        user_token: str | None = None,
    ):
        api_url = (
            f"{self.url}/v1/projects?is_regional=false&territory_id={territory_id}"
            f"&page=1&page_size=10000"
        )
        logger.info(f"Fetching list of projects for the territory from API: {api_url}")
        headers = await self._auth_headers(actor_user_id=actor_user_id, user_token=user_token)
        json_data = await self.handler.request("GET", api_url, session=self.session, headers=headers)
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

    async def get_context_territories(
        self,
        territory_id,
        project_id,
        actor_user_id: str | None = None,
    ):
        context_list = await self.get_context_ids(
            territory_id,
            project_id,
            actor_user_id=actor_user_id,
        )
        logger.info(f"Starting collection of territories for project {project_id}")
        all_context_territories = []
        chunk_size = 15
        for i in range(0, len(context_list), chunk_size):
            chunk = context_list[i:i + chunk_size]
            tasks = [
                self.get_territory(context_territory, actor_user_id=actor_user_id)
                for context_territory in chunk
            ]
            chunk_results = await asyncio.gather(*tasks)
            all_context_territories.extend(chunk_results)
        context_territories = pd.concat(all_context_territories)
        logger.info(f"Context territories for project {project_id} successfully collected")
        return context_territories
    
    async def get_social_groups(self, actor_user_id: str | None = None):
        api_url = (
            f"{self.url}/v1/social_groups"
        )
        headers = await self._auth_headers(actor_user_id=actor_user_id)
        results = await self.handler.request("GET", api_url, session=self.session, headers=headers)
        social_groups = pd.DataFrame(results)
        return social_groups

    async def get_services_for_groups(self, actor_user_id: str | None = None):
        social_groups = await self.get_social_groups(actor_user_id=actor_user_id)
        tasks = []
        for _, group in social_groups.iterrows():
            group_id = group['soc_group_id']
            api_url = f"{self.url}/v1/social_groups/{group_id}"
            logger.info(f"Prepared fetch task for social group {group_id}: {api_url}")
            headers = await self._auth_headers(actor_user_id=actor_user_id)
            coro = self.handler.request("GET", api_url, session=self.session, headers=headers)
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
        
    async def get_service_mapping(self, actor_user_id: str | None = None):
        api_url = f"{self.url}/v1/service_types"
        logger.info(f"Fetching list of projects for the territory from API: {api_url}")
        headers = await self._auth_headers(actor_user_id=actor_user_id)
        json_data = await self.handler.request("GET", api_url, session=self.session, headers=headers)
        logger.info(f"Service mapping successfully fetched from API.")
        service_mapping = pd.DataFrame([{"id": x["service_type_id"], "name": x["name"], "category": x['infrastructure_type']} for x in json_data])
        return service_mapping


urban_db_api = UrbanDBAPI(config)
