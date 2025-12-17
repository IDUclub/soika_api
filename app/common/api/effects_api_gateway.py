from fastapi import HTTPException
from loguru import logger
import pandas as pd
import aiohttp
from iduconfig import Config
from app.common.api.api_error_handler import APIHandler
from app.dependencies import config

class EffectsAPI:
    def __init__(self, config: Config):
        self.url = config.get("Effects_API")
        self.session = None
        self.handler = None
        self.config = config

    async def init(self):
        self.session = aiohttp.ClientSession()
        self.handler = APIHandler()

    async def close(self):
        if self.session:
            await self.handler.close_session(self.session)
            self.session = None
            logger.info("EffectsAPI session closed.")

    async def get_evaluated_territories_effects(self, scenario_id: int, token: str):
        api_url = (
            f"{self.url}/tasks/get_provisions/{scenario_id}"
        )
        headers = {"Authorization": f"Bearer {token}"}
        logger.info(f"GET {api_url}")

        try:
            json_data = await self.handler.request(
                "GET",
                api_url,
                session=self.session,
                headers=headers,
            )
            logger.success(f"Fetched effects for scenario {scenario_id}")
            result = pd.DataFrame(json_data)
            result.reset_index(inplace=True)
            result.columns = ["name_en", "before", "after"]
            after_is_empty = result["after"].isna().all()
            cols = ["before", "after"]
            result[cols] = result[cols].fillna(0)
            result["delta"] = 0 if after_is_empty else (result["after"] - result["before"]).round(2)
            return result

        except HTTPException as e:
            if e.status_code == 404:
                logger.warning(f"No data for {scenario_id}, triggering evaluation")

                eval_url = f"{self.url}/tasks/territory_transformation?force=false&scenario_id={scenario_id}"
                await self.handler.request(
                    "POST",
                    eval_url,
                    session=self.session,
                    headers=headers,
                    params={"method": "territory_transformation", "scenario_id": scenario_id},
                )
                logger.success(f"Triggered evaluation for scenario {scenario_id}")
                return None

            logger.exception(f"Error fetching effects for {scenario_id}")
            raise

        except Exception:
            logger.exception(f"Unexpected error for scenario {scenario_id}")
            raise

    async def get_services_ids(self, scenario_id: int, token: str):
        api_url = (
            f"{self.url}/tasks/get_service_types?scenario_id={scenario_id}&method=territory_transformation"
        )
        headers = {"Authorization": f"Bearer {token}"}
        logger.info(f"GET {api_url}")

        try:
            json_data = await self.handler.request(
                "GET",
                api_url,
                session=self.session,
                headers=headers,
            )
            logger.success(f"Fetched effects for scenario {scenario_id}")
            service_ids = pd.DataFrame(json_data["before"])
            service_ids.columns = ['id', 'name_en']
            return service_ids
        except Exception:
            raise


effects_api = EffectsAPI(config)