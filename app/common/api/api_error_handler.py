import aiohttp
from loguru import logger
from app.common.exceptions.http_exception_wrapper import http_exception

class APIHandler:
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session

    async def request(self, method: str, url: str, **kwargs):
        logger.info(f"Making {method} request to URL: {url}")
        async with self.session.request(method, url, **kwargs) as response:
            if response.status in (200, 201):
                try:
                    data = await response.json()
                    return data
                except Exception as e:
                    logger.error(f"JSON decode error from {url}: {e}")
                    raise http_exception(
                        response.status, "Invalid JSON response", url
                    )
            else:
                logger.error(f"Request to {url} failed with status {response.status}")
                raise http_exception(
                    response.status,
                    f"Request failed with status: {response.status}",
                    url,
                )

    async def close(self):
        if self.session:
            await self.session.close()
