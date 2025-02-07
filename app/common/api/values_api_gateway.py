import aiohttp
from loguru import logger
import json
import pandas as pd
import asyncio

from app.common.config import config
from app.common.exceptions.http_exception_wrapper import http_exception

class ValuesAPI:
    def __init__(self):
        self.url = config.get("Values_API")

    async def get_value_data(self, territory_id: int):
        """
        Collecting values data for region from Values_API.
        """
        api_url = f"{self.url}/regions/values_identities?territory_id={territory_id}"
        logger.info(f"Fetching values data from api: {api_url}")

        async with aiohttp.ClientSession() as session:
            async with session.get(api_url) as response:
                if response.status == 200:
                    json_data = await response.json()
                    logger.info(f"Values data for territory_id {territory_id} successfully fetched from API.")
                    df_values = pd.DataFrame.from_dict(json_data, orient='index')
                    df_melted = df_values.stack().reset_index()
                    df_melted.columns = ['social_group', 'value', 'indicator']
                    df_melted['social_value'] = df_melted['value'] + '/' + df_melted['social_group']
                    df_melted.set_index(df_melted['social_value'], inplace=True)
                    return df_melted['indicator']
                else:
                    logger.error(f"Failed to fetch values data, status code: {response.status}")
                    raise http_exception(response.status, f"Failed to fetch values data", str(response.url))
       
values_api = ValuesAPI()