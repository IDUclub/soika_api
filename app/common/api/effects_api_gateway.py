from loguru import logger
import pandas as pd
import requests

from app.common.config import config

class EffectsAPI:
    def __init__(self):
        self.url = config.get("Effects_API")
    async def get_evaluated_territories(self, scenario_id:int, token):
        api_url = f"{self.url}/effects/provision_data?project_scenario_id={scenario_id}&scale_type=Контекст"
        logger.info(f"Collecting scenario effects from API: {api_url}")
        response = requests.get(api_url, headers={'Authorization': f'Bearer {token}'})
        response.raise_for_status()
        json_data = response.json()
        logger.info(f"Effects for context for scenario {scenario_id} successfully fetched from API.")
        effects = pd.DataFrame(json_data)
        return effects
       
effects_api = EffectsAPI()