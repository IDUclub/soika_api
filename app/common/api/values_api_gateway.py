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
                    logger.info(
                        f"Values data for territory_id {territory_id} successfully fetched from API."
                    )
                    df_values = pd.DataFrame.from_dict(json_data, orient="index")
                    df_melted = df_values.stack().reset_index()
                    df_melted.columns = ["social_group", "value", "indicator"]
                    df_melted["indicator"] = df_melted["indicator"].map(lambda x: x[0])
                    df_melted["social_value"] = (
                        df_melted["value"] + "/" + df_melted["social_group"]
                    )
                    df_melted.set_index("social_value", inplace=True)

                    if not pd.api.types.is_numeric_dtype(df_melted["indicator"]):
                        error_message = f"Failed to parse valued_identities data: unexpected type {df_melted['indicator'].dtype}"
                        logger.error(error_message)
                        raise http_exception(
                            response.status, error_message, str(response.url)
                        )

                    return df_melted["indicator"]
                else:
                    error_message = (
                        f"Failed to fetch values data, status code: {response.status}"
                    )
                    logger.error(error_message)
                    raise http_exception(
                        response.status, error_message, str(response.url)
                    )


values_api = ValuesAPI()
