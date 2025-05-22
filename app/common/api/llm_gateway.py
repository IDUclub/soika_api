from fastapi import HTTPException

import aiohttp
import ssl
from loguru import logger
from iduconfig import Config
from app.dependencies import config
from app.common.api.api_error_handler import APIHandler
import requests


class LLMGateway:
    """
    Абстракция для общения с LLM-сервисом через защищённое HTTP-соединение.
    """
    def __init__(self, config: Config):
        self.url = config.get("Models_API")
        self.client_cert = config.get("GPU_CLIENT_CERTIFICATE")
        self.client_key = config.get("GPU_CLIENT_KEY")
        self.ca_cert = config.get("GPU_CERTIFICATE")
        self.session: aiohttp.ClientSession | None = None
        self.handler = APIHandler()

    async def init(self) -> None:
        """
        Инициализирует aiohttp-клиент с TLS-атрибуцией.
        """
        ssl_ctx = ssl.create_default_context(cafile=self.ca_cert)
        ssl_ctx.load_cert_chain(certfile=self.client_cert, keyfile=self.client_key)
        connector = aiohttp.TCPConnector(ssl=ssl_ctx)
        self.session = aiohttp.ClientSession(connector=connector)
        logger.info("LLMGateway session initialized.")

    async def close(self) -> None:
        """
        Закрывает клиентскую сессию.
        """
        if self.session:
            await self.handler.close_session(self.session)
            self.session = None
            logger.info("LLMGateway session closed.")

    async def get(self, extra_url: str, params: dict | None = None) -> dict:
        # если кто-то вдруг передал строку вместо dict —
        # горячо принудим это в нужное
        if isinstance(params, str):
            params = {'text': params}

        if self.session is None:
            await self.init()

        endpoint = f"{self.url.rstrip('/')}/{extra_url.lstrip('/')}"
        try:
            async with self.session.get(endpoint, params=params) as response:
                # если ответ не 2xx, это бросит ClientResponseError
                response.raise_for_status()
                return await response.json()

        except aiohttp.ClientResponseError as e:
            # получаем тело ошибки
            txt = await response.text()
            logger.error("LLM API returned %s: %s", e.status, txt)
            raise HTTPException(
                status_code=e.status,
                detail=f"LLM service error: {txt}"
            )

        except Exception as e:
            # что-то совсем неожиданное
            logger.exception("Unexpected error calling LLM API")
            raise HTTPException(
                status_code=500,
                detail=f"LLM gateway failure: {e}"
            )

    async def post(
        self,
        extra_url: str,
        json: dict | None = None,
    ) -> dict:
        if self.session is None:
            await self.init()

        endpoint = f"{self.url.rstrip('/')}/{extra_url.lstrip('/')}"
        try:
            async with self.session.post(
                endpoint,
                json=json
            ) as resp:
                resp.raise_for_status()
                return await resp.json()

        except aiohttp.ClientConnectionError as e:
            logger.error("LLM API POST %s connection error: %s", endpoint, e)
            raise HTTPException(502, detail=f"Cannot connect to LLM service: {e}")

        except Exception as e:
            logger.exception("Unexpected error in LLMGateway.post")
            raise HTTPException(status_code=500, detail=str(e))

llm_gateway = LLMGateway(config)
