import aiohttp
from loguru import logger
from app.common.exceptions.http_exception_wrapper import http_exception

class APIHandler:
    async def request(self, method: str, url: str, session: aiohttp.ClientSession, **kwargs):
        logger.info(f"Making {method} request to URL: {url}")
        async with session.request(method, url, **kwargs) as response:
            if response.status in (200, 201):
                try:
                    data = await response.json()
                    return data
                except Exception as e:
                    logger.error(f"JSON decode error from {url}: {e}")
                    raise http_exception(
                        500,
                        "Invalid JSON response",
                        url,
                        str(e)
                    )
            else:
                logger.error(f"Request to {url} failed with status {response.status}")
                detail = await response.text()
                raise http_exception(
                    response.status,
                    f"Request failed with status: {response.status}",
                    url,
                    detail
                )

    @staticmethod
    async def close_session(session: aiohttp.ClientSession):
        await session.close()