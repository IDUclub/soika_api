import asyncio
import requests
import pandas as pd
from loguru import logger
from app.common.db.database import (
    Message,
    Indicator,
    MessageIndicator
)
from app.common.db.db_engine import database
from sqlalchemy import select, delete
from app.common.exceptions.http_exception_wrapper import http_exception
from app.preprocessing.modules import utils
from iduconfig import Config
from app.dependencies import config

class IndicatorsCalculation:
    def __init__(self, config: Config):
        self.config = config
        self.url = config.get("GPU_URL")
        self.client_cert = config.get("GPU_CLIENT_CERTIFICATE")
        self.client_key = config.get("GPU_CLIENT_KEY")
        self.ca_cert = config.get("GPU_CERTIFICATE")

    def construct_prompt(self, context):
        """
        Формирует промпт для модели на основе текста обращения.
        """
        logger.debug("Начало формирования промпта. Исходный context: ", context)
        context_str = "\n".join(context) if isinstance(context, list) else str(context)
        prompt = f"""
            Найди обсуждение показателей в тексте {context_str}. 
            Строительство - обращение упоминает строительство новых объектов, реновацию или реконструкцию, открытие объектов, постройку.
            Снос - обращение упоминает уничтожение объектов, их разрушение, повреждение. Любая утрата объекта или его части умышленным образом. 
            Обеспеченность - обращение упоминает то, насколько сервис(объект) загружен жителями. Признаки чрезмерной загруженности - очереди, нехватка мест, нехватка персонала.
            Доступность - обращение упоминает сложности с достижением объекта или сервиса в разумное время. Слишком долго или сложно добираться, слишком большое расстояние до ближайшего объекта. 
            В ответе должен быть список из упомянутых показателей в формате [ind1, ind2, ind3]. Если показатели не обсуждаются, пиши []. Не пиши ничего, кроме списка. Нужен ответ правильного формата.
            """
        logger.debug("Сформированный промпт: ", prompt)
        return prompt

    async def describe_async(self, context):
        """
        Асинхронно отправляет запрос с сформированным промптом.
        """
        prompt = indicators_calculation.construct_prompt(context)
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "deepseek-r1:32b",
            "temperature": 0.0,
            "prompt": prompt,
            "stream": False,
            "think":False
        }

        def sync_request():
            try:
                response = requests.post(
                    self.url,
                    headers=headers,
                    json=data,
                    cert=(self.client_cert, self.client_key),
                    verify=self.ca_cert,
                )
                if response.status_code == 200:
                    response_json = response.json()
                    return response_json.get("response", "")
                else:
                    logger.error(
                        "Ошибка запроса: , ответ: ",
                        response.status_code,
                        response.text,
                    )
                    return None
            except requests.exceptions.RequestException as e:
                logger.error("Ошибка соединения при запросе: ", e)
                return None

        result = await asyncio.to_thread(sync_request)
        return result

    async def process_find_indicators(self, items):
        """
        Обрабатывает список словарей с ключом 'context' и возвращает список результатов.
        """
        tasks = [indicators_calculation.describe_async(item["context"]) for item in items]
        results = await utils.gather_with_progress(tasks, description="В процессе")
        logger.info("Завершена обработка описаний.")
        return results

    @staticmethod
    def parse_indicator_response(s):
        """
        Парсит строку ответа в список показателей.
        """
        s = s.strip()
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1].strip()
        else:
            inner = s
        if not inner:
            return []
        return [word.strip().capitalize() for word in inner.split(",") if word.strip()]

    @staticmethod
    def process_indicators(indicators):
            return [indicators_calculation.parse_indicator_response(x) for x in indicators]
    
    async def get_indicators(self, df):
        """
        Принимает DataFrame с колонкой 'text', обрабатывает запросы и возвращает список результатов.
        """
        def build_items(df):
            return [{"context": row["text"]} for _, row in df.iterrows()]
        items = await asyncio.to_thread(build_items, df)

        indicators = await indicators_calculation.process_find_indicators(items)
        if not indicators or all(d is None for d in indicators):
            raise http_exception(
                status_code=400,
                msg="Ошибка соединения при запросах",
                input_data=indicators,
                detail="Проверьте корректность подключения к LLM"
            )
        processed_indicators = await asyncio.to_thread(indicators_calculation.process_indicators, indicators)
        return processed_indicators

    async def extract_indicators(self, top: int = None) -> dict:
        """
        1) Находит сообщения is_processed=False (огранич. top, если нужно).
        2) Для каждого собирает DF (message_id, text).
        3) Вызывает get_indicators(df) -> список списков индикаторов.
        4) Для каждого сообщения -> индикаторы -> сохраняем в message_indicator.
        5) Возвращаем инфо о количестве связей.
        """
        async with database.session() as session:
            query = select(Message).where(Message.is_processed == False)
            if top is not None and top > 0:
                query = query.limit(top)

            result = await session.execute(query)
            messages = result.scalars().all()

            if not messages:
                return {
                    "detail": "No unprocessed messages found.",
                    "processed_messages": 0,
                }
            data = []
            for msg in messages:
                data.append({"message_id": msg.message_id, "text": msg.text})

            df = await asyncio.to_thread(pd.DataFrame, data)

            indicators_list = await self.get_indicators(df)
            total_links_created = 0

            for i, row in df.iterrows():
                mid = row["message_id"]
                found_inds = indicators_list[i]
                if not found_inds:
                    continue

                for ind_name in found_inds:
                    stmt = select(Indicator).where(Indicator.name == ind_name)
                    existing = await session.execute(stmt)
                    indicator_obj = existing.scalar_one_or_none()

                    if not indicator_obj:
                        indicator_obj = Indicator(name=ind_name)
                        session.add(indicator_obj)
                        await session.flush()

                    link_stmt = select(MessageIndicator).where(
                        MessageIndicator.message_id == mid,
                        MessageIndicator.indicator_id == indicator_obj.indicator_id,
                    )
                    link_res = await session.execute(link_stmt)
                    link_exists = link_res.scalar_one_or_none()

                    if not link_exists:
                        link = MessageIndicator(
                            message_id=mid, indicator_id=indicator_obj.indicator_id
                        )
                        session.add(link)
                        total_links_created += 1

            await session.commit()

        return {
            "detail": f"Created {total_links_created} message_indicator records.",
            "processed_messages": len(messages),
        }
    
    @staticmethod
    async def get_all_indicators():
        async with database.session() as session:
            result = await session.execute(select(Indicator))
            indicators = result.scalars().all()
        indicators_list = [{"indicator_id": i.indicator_id, "name": i.name} for i in indicators]
        return {"indicators": indicators_list}

    @staticmethod
    async def get_all_message_indicator_pairs():
        async with database.session() as session:
            result = await session.execute(select(MessageIndicator))
            indicators = result.scalars().all()
        indicators_list = [{"message_id": i.message_id, "indicator_id": i.indicator_id} for i in indicators]
        return {"message_indicator_pairs": indicators_list}

    @staticmethod
    async def create_indicator_func(payload):
        async with database.session() as session:
            new_indicator = Indicator(name=payload.name)
            session.add(new_indicator)
            await session.commit()
            await session.refresh(new_indicator)
        return {"indicator_id": new_indicator.indicator_id, "name": new_indicator.name}

    @staticmethod
    async def extract_indicators_func(top: int = None):
        result = await indicators_calculation.extract_indicators(top=top)
        return result

    @staticmethod
    async def delete_all_indicators_func():
        async with database.session() as session:
            await session.execute(delete(Indicator))
            await session.commit()
        return {"detail": "All indicators deleted"}

indicators_calculation = IndicatorsCalculation(config)