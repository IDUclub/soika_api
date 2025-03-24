from loguru import logger
import pandas as pd
import aiohttp
from iduconfig import Config
from app.common.api.api_error_handler import APIHandler

config = Config()

class EffectsAPI:
    def __init__(self):
        self.url = config.get("Effects_API")
        self.session = None
        self.handler = None

    async def init(self):
        self.session = aiohttp.ClientSession()
        self.handler = APIHandler(self.session)

    async def get_evaluated_territories(self, scenario_id: int, token: str):
        api_url = f"{self.url}/effects/provision_data?project_scenario_id={scenario_id}&scale_type=Контекст"
        logger.info(f"Collecting scenario effects from API: {api_url}")

        headers = {'Authorization': f'Bearer {token}'}
        json_data = await self.handler.request("GET", api_url, headers=headers)
        logger.info(
            f"Effects for context for scenario {scenario_id} successfully fetched from API."
        )
        effects = pd.DataFrame(json_data)
        return effects

effects_api = EffectsAPI()