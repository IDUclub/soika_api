from app.common.config import config
from app.common.db.database import (
    database,
    Territory,
    Group,
    Message,
    Emotion,
    NamedObject,
    MessageNamedObject,
    Territory,
    GroupTerritory,
    Indicator,
    MessageIndicator,
    Service,
    MessageService,
)
from sqlalchemy import select, func
import asyncio
import numpy as np
import requests
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
import pandas as pd
import geopandas as gpd
from soika import Geocoder, VKParser
import torch
from flair.models import SequenceTagger
from app.risk_calculation.logic.constants import CONSTANTS
import re
from rapidfuzz import fuzz
from tqdm import tqdm
import osmnx as ox
import ast
from shapely.geometry import Point
from shapely import wkt
from fastapi import UploadFile, HTTPException
import csv
from io import StringIO
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# TODO: разбить методы соответственно роутерам
class Preprocessing:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._classification_model = None
        # TODO: переобучить модель с flair на pytorch transformers
        self._ner_model = None
        logger.info(f"CUDA avaliable: {torch.cuda.is_available()}")

    async def init_models(self):
        """
        Асинхронная инициализация двух моделей:
        - Модель для классификации эмоций через Transformers pipeline.
        - Модель для извлечения адресов через Flair SequenceTagger.
        """
        loop = asyncio.get_event_loop()
        classification_pipeline = "text-classification"
        classification_model_name = "Sandrro/emotions_classificator_v4"
        ner_model_name = "Geor111y/flair-ner-addresses-extractor"

        logger.info(
            f"Launching classification model {classification_model_name} for {classification_pipeline}"
        )
        self._classification_model = await loop.run_in_executor(
            self.executor,
            lambda: pipeline(
                classification_pipeline,
                model=classification_model_name,
                truncation=True,
                max_length=512,
            ),  # TODO: нужно будет наладить обработку по частям
        )
        logger.info(
            f"Launching NER model {ner_model_name} with Flair SequenceTagger"
        )
        self._ner_model = await loop.run_in_executor(
            self.executor, lambda: SequenceTagger.load(ner_model_name)
        )

    def get_classification_model(self):
        return self._classification_model

    def get_ner_model(self):
        return self._ner_model

    @staticmethod
    async def get_groups_by_territory_id(territory_id: int):
        async with database.session() as session:
            territory = await session.get(Territory, territory_id)
            if not territory:
                raise HTTPException(
                    status_code=404,
                    detail=f"Territory with id={territory_id} not found",
                )
            territory_name = territory.name
            query = select(Group).where(Group.matched_territory == territory_name)
            result = await session.execute(query)
            groups = result.scalars().all()

        return groups

    async def create_territory(self, payload: Territory):
        """
        Создаёт новую запись в таблице territory.
        Возвращает объект SQLAlchemy-модели Territory.
        """
        async with database.session() as session:
            new_territory = Territory(
                territory_id=payload.territory_id,
                name=payload.name,
                matched_territory=payload.matched_territory,
            )
            session.add(new_territory)
            await session.commit()

            return new_territory

    async def search_vk_groups(
        self, territory_id: int, sort: int = 4, count: int = 20, version: str = "5.131"
    ) -> str:
        async with database.session() as session:
            territory = await session.get(Territory, territory_id)
            if not territory:
                raise HTTPException(
                    status_code=404,
                    detail=f"Territory with id={territory_id} not found",
                )

            territory_name = territory.name

        group_access_key = config.get("VK_GROUP_ACCESS_KEY")
        params = {
            "q": territory_name,
            "sort": sort,
            "count": count,
            "access_token": group_access_key,
            "v": version,
        }

        response = await asyncio.to_thread(
            requests.get, "https://api.vk.com/method/groups.search", params=params
        )
        data = response.json()

        df = pd.DataFrame(data["response"]["items"])[["id", "name", "screen_name"]]
        df.rename(
            columns={"screen_name": "group_domain", "id": "group_id"}, inplace=True
        )
        df["matched_territory"] = territory_name

        records = df.to_dict("records")
        async with database.session() as session:
            for record in records:
                group_obj = Group(
                    group_id=record["group_id"],
                    name=record["name"],
                    group_domain=record["group_domain"],
                    matched_territory=record["matched_territory"],
                )
                session.add(group_obj)
            await session.commit()

        return territory_name

    async def add_messages(self, file: UploadFile):
        """
        Обработка CSV-файла для добавления сообщений в базу данных.
        """
        logger.info("Начало обработки CSV-файла для добавления сообщений.")

        # Костыль: Добавляем искусственное сообщение с message_id = 1, если его ещё нет
        async with database.session() as session:
            result = await session.execute(select(Message).where(Message.message_id == 1))
            dummy = result.scalars().first()
            if not dummy:
                dummy_message = Message(
                    message_id=1,
                    text="Искусственное сообщение - dummy record",
                    date=datetime.now(),
                    views=0,
                    likes=0,
                    reposts=0,
                    type="post",
                    parent_message_id=1,
                    group_id=None,
                    emotion_id=None,
                    score=None,
                    geometry=None,
                    location=None,
                    is_processed=True,
                )
                session.add(dummy_message)
                await session.commit()
                logger.info("Искусственное сообщение добавлено с message_id=1.")

        content = await file.read()
        decoded = content.decode("utf-8")
        csv_reader = csv.DictReader(StringIO(decoded))
        messages = []
        ru_service_names = CONSTANTS.json["ru_service_names"]

        async with database.session() as session:
            for row in csv_reader:
                emotion_name = row.get("emotion")
                result = await session.execute(select(Emotion).where(Emotion.name == emotion_name))
                emotion_obj = result.scalars().first()
                if not emotion_obj:
                    logger.error(f"Эмоция '{emotion_name}' не найдена в базе данных")
                    raise Exception(f"Эмоция '{emotion_name}' не найдена в базе данных")
                emotion_id = emotion_obj.emotion_id

                geom_str = row.get("geometry")
                if geom_str:
                    try:
                        if geom_str.strip().upper().startswith("POINT"):
                            from shapely import wkt
                            point = wkt.loads(geom_str)
                        else:
                            x_str, y_str = geom_str.split(",")
                            point = Point(float(x_str.strip()), float(y_str.strip()))
                        ewkt_geometry = self.to_ewkt(point)
                    except Exception as e:
                        logger.error(f"Неверный формат геометрии '{geom_str}'. Ошибка: {e}")
                        raise Exception(f"Неверный формат геометрии '{geom_str}'. Ошибка: {e}")
                else:
                    logger.error("Отсутствует значение геометрии")
                    raise Exception("Отсутствует значение геометрии")
                
                territory_id = None
                territory_name = row.get("territory_name")
                if territory_name:
                    result = await session.execute(select(Territory).where(Territory.name == territory_name))
                    territory_obj = result.scalars().first()
                    if not territory_obj:
                        territory_obj = Territory(name=territory_name)
                        session.add(territory_obj)
                        await session.flush()
                        logger.info(f"Создана новая территория: {territory_name}")
                    territory_id = territory_obj.territory_id

                group_id = None
                group_name = row.get("group_name")
                if group_name:
                    result = await session.execute(select(Group).where(Group.name == group_name))
                    group_obj = result.scalars().first()
                    if not group_obj:
                        matched_territory = row.get("territory_name")
                        group_domain = row.get("group_domain")
                        group_obj = Group(name=group_name, group_domain=group_domain, matched_territory=matched_territory)
                        session.add(group_obj)
                        await session.flush()
                        logger.info(f"Создана новая группа: {group_name}")
                    group_id = group_obj.group_id

                message = Message(
                    text=row.get("text"),
                    date=datetime.fromisoformat(row.get("date")) if row.get("date") else None,
                    views=int(row.get("views")) if row.get("views") else None,
                    likes=int(row.get("likes")) if row.get("likes") else None,
                    reposts=int(row.get("reposts")) if row.get("reposts") else None,
                    type=row.get("type"),
                    parent_message_id=1,  # Продолжение костыля - дальше должно вести на нормальный айдишник
                    group_id=group_id,
                    emotion_id=emotion_id,
                    score=float(row.get("score")) if row.get("score") else None,
                    geometry=ewkt_geometry,
                    location=row.get("location"),
                    is_processed=row.get("is_processed").strip().lower() in ["true", "1"]
                        if row.get("is_processed") else False,
                )
                session.add(message)
                await session.flush()

                message.parent_message_id = message.message_id
                await session.flush()

                services_field = row.get("services")
                if services_field:
                    services_list = [s.strip() for s in services_field.split(",") if s.strip()]
                    services_list = self.replace_service_names(services_list, ru_service_names)
                    services_list = list(dict.fromkeys(services_list))
                    with session.no_autoflush:
                        for service_name in services_list:
                            result = await session.execute(
                                select(Service).where(Service.name == service_name)
                            )
                            service_obj = result.scalars().first()
                            if not service_obj:
                                service_obj = Service(name=service_name)
                                session.add(service_obj)
                                await session.flush()
                                logger.info(f"Сервис '{service_name}' добавлен в базу данных")
                            msg_service = MessageService(
                                message_id=message.message_id, service_id=service_obj.service_id
                            )
                            session.add(msg_service)

                indicators_field = row.get("indicators")
                if indicators_field:
                    indicators_list = [s.strip() for s in indicators_field.split(",") if s.strip()]
                    indicators_list = list(dict.fromkeys(indicators_list))
                    with session.no_autoflush:
                        for indicator_name in indicators_list:
                            result = await session.execute(
                                select(Indicator).where(Indicator.name == indicator_name)
                            )
                            indicator_obj = result.scalars().first()
                            if not indicator_obj:
                                logger.error(f"Индикатор '{indicator_name}' не найден в базе данных")
                                raise Exception(f"Индикатор '{indicator_name}' не найден в базе данных")
                            result = await session.execute(
                                select(MessageIndicator).where(
                                    MessageIndicator.message_id == message.message_id,
                                    MessageIndicator.indicator_id == indicator_obj.indicator_id
                                )
                            )
                            existing = result.scalars().first()
                            if existing:
                                logger.warning(
                                    f"Связь для индикатора {indicator_obj.indicator_id} уже существует для сообщения {message.message_id}. Пропускаем."
                                )
                                continue
                            msg_indicator = MessageIndicator(
                                message_id=message.message_id, indicator_id=indicator_obj.indicator_id
                            )
                            session.add(msg_indicator)

                if group_id and territory_id:
                    result = await session.execute(
                        select(GroupTerritory).where(
                            GroupTerritory.group_id == group_id,
                            GroupTerritory.territory_id == territory_id
                        )
                    )
                    group_territory_obj = result.scalars().first()
                    if not group_territory_obj:
                        group_territory_obj = GroupTerritory(group_id=group_id, territory_id=territory_id)
                        session.add(group_territory_obj)
                messages.append(message)
            await session.commit()
        logger.info(f"Обработка файла завершена. Добавлено сообщений: {len(messages)}")
        return messages


    @staticmethod
    async def get_latest_message_date_by_territory_id(territory_id: int):
        """
        Возвращает самую актуальную дату (max(Message.date)) по всем сообщениям,
        связанным с группами, которые принадлежат указанной территории.
        Если сообщений нет, вернёт None.
        """
        async with database.session() as session:
            query = (
                select(func.max(Message.date))
                .select_from(Territory)
                .join(GroupTerritory, Territory.territory_id == GroupTerritory.territory_id)
                .join(Group, Group.group_id == GroupTerritory.group_id)
                .join(Message, Message.group_id == Group.group_id)
                .where(Territory.territory_id == territory_id)
            )
            result = await session.execute(query)
            latest_date = result.scalar()
        return latest_date


    async def parse_VK_texts(self, territory_id: int, cutoff_date: str = None, limit: int = None):
        """
        Получает сообщения из ВК для групп, связанных с territory_id.
        """
        if not cutoff_date:
            latest_date = await self.get_latest_message_date_by_territory_id(territory_id)
            if latest_date:
                next_day = latest_date + timedelta(days=1)
                cutoff_date = next_day.strftime("%Y-%m-%d")
            else:
                two_years_ago = datetime.now() - relativedelta(years=2)
                cutoff_date = two_years_ago.strftime("%Y-%m-%d")

        access_key = config.get("VK_ACCESS_KEY")
        parser = VKParser()
        groups = await self.get_groups_by_territory_id(territory_id)
        all_messages_data = []
        total_messages_count = 0

        logger.info(
            f"Starting to parse VK texts for territory_id={territory_id}, "
            f"cutoff_date={cutoff_date}, found {len(groups)} groups."
        )

        for group in groups:
            if limit is not None and total_messages_count >= limit:
                logger.info("Global message limit reached. Stopping further processing.")
                break

            logger.info(
                f"Processing group_id={group.group_id}, domain={group.group_domain}..."
            )

            df = parser.run_parser(
                domain=group.group_domain,
                access_token=access_key,
                cutoff_date=cutoff_date,
            )

            if df.empty:
                logger.info(f"No new messages for group_id={group.group_id}. Skipping.")
                continue

            if limit is not None:
                remaining = limit - total_messages_count
                df = df.head(remaining)

            df["group_id"] = group.group_id
            df = df.replace({np.nan: None})
            df["date"] = pd.to_datetime(df["date"], utc=True)
            messages_data = df.to_dict("records")

            for message in messages_data:
                original_vk_id = message.pop("id")
                message["views"] = message.pop("views.count")
                message["likes"] = message.pop("likes.count")
                message["reposts"] = message.pop("reposts.count")
                message.pop("parents_stack", None)  # не используем parents_stack
                message["parent_message_id"] = 1
                message["score"] = None
                message["location"] = None
                message["geometry"] = None
                message["emotion_id"] = None
                message["is_processed"] = False

            async with database.session() as session:
                for msg in messages_data:
                    message_obj = Message(
                        text=msg["text"],
                        date=msg["date"],
                        views=msg["views"],
                        likes=msg["likes"],
                        reposts=msg["reposts"],
                        type=msg["type"],
                        parent_message_id=msg["parent_message_id"],
                        group_id=msg["group_id"],
                        emotion_id=msg["emotion_id"],
                        score=msg["score"],
                        geometry=msg["geometry"],
                        location=msg["location"],
                        is_processed=msg["is_processed"],
                    )
                    session.add(message_obj)

                await session.commit()

            logger.info(
                f"Committed {len(messages_data)} messages to DB "
                f"for group_id={group.group_id}."
            )
            total_messages_count += len(messages_data)
            all_messages_data.extend(messages_data)

        logger.info(
            f"Finished parsing territory_id={territory_id}. "
            f"Total messages processed: {total_messages_count}."
        )
        return all_messages_data

    async def classify_emotion(self, text: str):
        """
        Асинхронная функция для получения предсказания по тексту с помощью модели классификации.
        """
        model = self.get_classification_model()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, model, text)
        return result[0]["label"]

    async def get_osm_id_by_territory_name(self, territory_name: str) -> int | None:
        """
        Ищет osm_id для указанной территории через osmnx.
        Возвращает int (osm_id) или None, если не удалось найти.
        """
        loop = asyncio.get_running_loop()

        def _sync_geocode(name: str) -> int | None:
            try:
                gdf = ox.geocode_to_gdf(name)
                if gdf.empty:
                    return None
                return int(gdf.iloc[0]["osm_id"])
            except Exception as e:
                logger.error(f"Error geocoding '{name}' via osmnx: {e}")
                return None

        osm_id = await loop.run_in_executor(None, _sync_geocode, territory_name)
        return osm_id

    @staticmethod
    def to_ewkt(geom: Point, srid: int = 4326) -> str:
        """
        Формируем строку вида SRID=4326;POINT(x y)
        """
        return f"SRID={srid};POINT({geom.x} {geom.y})"

    async def extract_addresses_for_unprocessed(
        self,
        device: str = "cpu",
        top: int = None,
        input_territory_name: str = None
    ) -> list[dict]:
        """
        1) Ищет все сообщения (Message), у которых is_processed=False (ограничиваем top, если задан).
        2) Одним запросом получает все группы (Group), нужные для этих сообщений.
        3) Одним запросом получает все территории (Territory), нужные для этих групп.
        4) Для территорий без osm_id вызывает get_osm_id_by_territory_name (osmnx).
        5) Группирует сообщения по osm_id, пакетно вызывает geocoder.run().
        6) Обновляет поля location, geometry, is_processed (если не NaN).
        7) Возвращает список словарей с результатами.
        """
        async with database.session() as session:
            msg_query = select(Message).where(Message.is_processed == False)
            if top is not None and top > 0:
                msg_query = msg_query.limit(top)
            msg_result = await session.execute(msg_query)
            messages = msg_result.scalars().all()

            if not messages:
                logger.info("No unprocessed messages found for address extraction.")
                return []

            group_ids = {m.group_id for m in messages if m.group_id is not None}
            if not group_ids:
                logger.info("No valid group_id found among unprocessed messages.")
                return []

            grp_query = select(Group).where(Group.group_id.in_(group_ids))
            grp_result = await session.execute(grp_query)
            groups = grp_result.scalars().all()
            group_map = {g.group_id: g for g in groups}

            territory_names = set()
            for g in groups:
                if g.matched_territory:
                    territory_names.add(g.matched_territory)

            if not territory_names:
                logger.info("No territory names found in groups.")
                return []

            terr_query = select(Territory).where(Territory.name.in_(territory_names))
            terr_result = await session.execute(terr_query)
            territories = terr_result.scalars().all()
            territory_map = {t.name: t for t in territories}

            for t in territories:
                if not getattr(t, "osm_id", None):
                    logger.info(f"No osm_id in DB for territory='{t.name}'. Trying osmnx...")
                    osm_id = await self.get_osm_id_by_territory_name(t.name)
                    if osm_id:
                        t.osm_id = osm_id
                        logger.info(f"OSM id {t.osm_id}")
            # await session.commit()

            messages_by_osm = {}
            for msg in messages:
                grp = group_map.get(msg.group_id)
                if not grp:
                    logger.warning(
                        f"Message {msg.message_id} has invalid group_id={msg.group_id}. Skipping."
                    )
                    continue

                territory_name = grp.matched_territory
                if not territory_name:
                    logger.warning(
                        f"Group {grp.group_id} has no matched_territory. Skipping message {msg.message_id}."
                    )
                    continue

                terr_obj = territory_map.get(territory_name)
                if not terr_obj:
                    logger.warning(
                        f"Territory '{territory_name}' not found in DB. Skipping msg={msg.message_id}."
                    )
                    continue

                osm_id = getattr(terr_obj, "osm_id", None)
                if not osm_id:
                    logger.warning(
                        f"Could not determine osm_id for territory='{territory_name}'. Skipping msg={msg.message_id}."
                    )
                    continue

                messages_by_osm.setdefault(osm_id, []).append(msg)

            updated_records = []

            for osm_key, msgs in messages_by_osm.items():
                df = pd.DataFrame({
                    "message_id": [m.message_id for m in msgs],
                    "text": [m.text for m in msgs],
                })
                if df.empty:
                    continue

                geocoder = Geocoder(
                    df=df,
                    osm_id=osm_key,
                    device=device,
                    model_path="Geor111y/flair-ner-addresses-extractor",
                    text_column_name="text",
                    city_tags={"admin_level": ["6"]},
                    territory_name=input_territory_name
                )
                result_gdf = geocoder.run(group_column=None, search_for_objects=False)

                if "message_id" not in result_gdf.columns:
                    logger.warning(
                        f"Geocoder result has no 'message_id' column. Skipping osm_id={osm_key}."
                    )
                    continue

                records = result_gdf.to_dict("records")
                msg_map = {m.message_id: m for m in msgs}

                for row in records:
                    mid = row["message_id"]
                    loc = row.get("Location")
                    geom_data = row.get("geometry")

                    if pd.isna(loc) or pd.isna(geom_data):
                        logger.debug(f"Skipping message_id={mid} due to NaN in location/geometry.")
                        continue

                    msg_obj = msg_map.get(mid)
                    if not msg_obj:
                        continue

                    msg_obj.location = loc
                    if isinstance(geom_data, Point):
                        msg_obj.geometry = self.to_ewkt(geom_data, srid=4326)
                    else:
                        msg_obj.geometry = None

                    msg_obj.is_processed = True

                    updated_records.append(
                        {
                            "message_id": mid,
                            "osm_id": osm_key,
                            "location": str(loc),
                            "geometry": msg_obj.geometry,
                        }
                    )

            await session.commit()

        logger.info(f"Batch extraction done. Updated {len(updated_records)} messages.")
        return updated_records

    @staticmethod
    def fuzzy_search(text, phrase, threshold=80):
        """
        Ищет фразу в тексте с использованием нестрогого сравнения.
        Возвращает True, если найден фрагмент текста, схожий с целевым по уровню similarity >= threshold.
        """
        text = text.lower()
        phrase = phrase.lower()
        words = text.split()
        phrase_words = phrase.split()
        n = len(phrase_words)
        if n == 0:
            return False
        for i in range(len(words) - n + 1):
            window = " ".join(words[i : i + n])
            if fuzz.ratio(phrase, window) >= threshold:
                return True
        return False

    @staticmethod
    def replace_service_names(services_list, ru_service_names):
        inverted = {eng: rus for rus, eng in ru_service_names.items()}
        return [inverted.get(service, service) for service in services_list]

    async def detect_services(self, text: str) -> list[str]:
        """
        Возвращает список сервисов (названий ключей) для одного текста.
        """
        detected_services = []
        text_lower = text.lower()
        service_keywords = CONSTANTS.json["service_keywords"]
        service_irrelevant_mentions = CONSTANTS.json["service_irrelevant_mentions"]
        services_priority_and_exact_keywords = CONSTANTS.json[
            "services_priority_and_exact_keywords"
        ]
        ru_service_names = CONSTANTS.json["ru_service_names"]

        for service, keywords in service_keywords.items():
            found = False
            for kw in keywords:
                if self.fuzzy_search(text_lower, kw):
                    found = True
                    break
            if not found:
                continue

            for irr in service_irrelevant_mentions.get(service, []):
                if self.fuzzy_search(text_lower, irr):
                    found = False
                    break
            if not found:
                continue

            if service in services_priority_and_exact_keywords:
                local_config = services_priority_and_exact_keywords[service]
                exact_found = any(
                    self.fuzzy_search(text_lower, ex_kw, threshold=85)
                    for ex_kw in local_config.get("exact_keywords", [])
                )
                if exact_found:
                    detected_services.append(service)
                    continue

                priority_over_found = any(
                    self.fuzzy_search(text_lower, po_kw, threshold=85)
                    for po_kw in local_config.get("priority_over", [])
                )
                if priority_over_found:
                    continue
                for excl in local_config.get("exclude_verbs", []):
                    if self.fuzzy_search(text_lower, excl, threshold=85):
                        found = False
                        break
                if not found:
                    continue

            detected_services.append(service)

        # TODO: добавить перевод для всех названий
        detected_services = self.replace_service_names(
            detected_services, ru_service_names
        )
        return detected_services

    async def extract_services_in_messages(self, top: int = None) -> dict:
        """
        1) Находит все сообщения is_processed=False (огранич. top, если нужно).
        2) Для каждого вызывает detect_services(text).
        3) По каждому найденному сервису - ищет/создаёт запись в service,
           затем создаёт запись в message_service.
        4) Возвращает инфо о кол-ве созданных связей.
        """
        async with database.session() as session:
            # 1. Выбираем сообщения
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

            total_links_created = 0

            for msg in messages:
                services_found = await self.detect_services(msg.text)
                if not services_found:
                    continue
                for serv_name in services_found:
                    stmt = select(Service).where(Service.name == serv_name)
                    existing = await session.execute(stmt)
                    service_obj = existing.scalar_one_or_none()

                    if not service_obj:
                        service_obj = Service(name=serv_name)
                        session.add(service_obj)
                        await session.flush()

                    link_stmt = select(MessageService).where(
                        MessageService.message_id == msg.message_id,
                        MessageService.service_id == service_obj.service_id,
                    )
                    link_res = await session.execute(link_stmt)
                    link_exists = link_res.scalar_one_or_none()

                    if not link_exists:
                        link = MessageService(
                            message_id=msg.message_id, service_id=service_obj.service_id
                        )
                        session.add(link)
                        total_links_created += 1

            await session.commit()

        return {
            "detail": f"Created {total_links_created} message_service records.",
            "processed_messages": len(messages),
        }

class NER_EXTRACTOR:
    def __init__(self):
        self.url = config.get("GPU_URL")
        self.client_cert = config.get("GPU_CLIENT_CERTIFICATE")
        self.client_key = config.get("GPU_CLIENT_KEY")
        self.ca_cert = config.get("GPU_CERTIFICATE")

    def construct_prompt(self, context):
        """
        Формирует промпт для модели на основе контекста.
        """
        logger.debug("Начало формирования промпта. Исходный context: %s", context)
        context_str = "\n".join(context) if isinstance(context, list) else str(context)
        dict_example = {
            "name": "Юбилейный",
            "notes": "Исторический ресторан на окраине города",
            "location": "Дубровка, Россия",
        }
        prompt = f"""
            Найди названия в тексте {context_str}. 
            Названия должны принадлежать объектам или организациям, которые физически представлены на территории.
            Приведи названия в начальную форму. Добавь как можно больше названий на основе контекста.
            Если в тексте нет упоминаний местоположения, это Ленинграская область.
            Преобразуй название, краткое описание на основе контекста и местоположение в словарь для поиска в OSM.
            Если названий нет, верни только пустой словарь.
            Словарь должен быть корректной формы.
            Пример итогового словаря: {dict_example}
            Если названий больше одного, сохрани словари элементами в списке через запятую [dict1, dict2, dict3]
            """
        logger.debug("Сформированный промпт: %s", prompt)
        return prompt

    async def describe_async(self, context):
        """
        Асинхронно отправляет запрос с сформированным промптом.
        """
        prompt = self.construct_prompt(context)
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "deepseek-r1:70b",
            "temperature": 0.1,
            "prompt": prompt,
            "stream": False,
        }

        def sync_request():
            logger.info("Отправка запроса", self.url)
            try:
                response = requests.post(
                    self.url,
                    headers=headers,
                    json=data,
                    cert=(self.client_cert, self.client_key),
                    verify=self.ca_cert,
                )
                if response.status_code == 200:
                    logger.info(
                        "Получен успешный ответ от модели.",
                        response.status_code,
                    )
                    response_json = response.json()
                    return response_json.get("response", "")
                else:
                    logger.error(
                        "Ошибка запроса",
                        response.status_code,
                        response.text,
                    )
                    return None
            except requests.exceptions.RequestException as e:
                logger.error("Ошибка соединения при запросе", e)
                return None

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, sync_request)
        return result

    async def process_write_descriptions(self, items):
        """
        Обрабатывает список элементов items (каждый со своим 'context') асинхронно.
        """
        logger.info("Начало обработки описаний.", len(items))
        tasks = [self.describe_async(item["context"]) for item in items]
        pbar = tqdm(total=len(tasks), desc="В процессе")

        async def run_task(task):
            result = await task
            pbar.update(1)
            return result

        results = await asyncio.gather(*(run_task(task) for task in tasks))
        pbar.close()
        logger.info("Завершена обработка описаний.")
        return results

    def split_extracted_data(self, df):
        """
        Разбивает содержимое столбца 'extracted_data' на два столбца: 'logic' и 'response'.
        """

        def extract_think(text):
            match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
            return match.group(1).strip() if match else ""

        def extract_response(text):
            text_without_think = re.sub(
                r"<think>.*?</think>", "", text, flags=re.DOTALL
            ).strip()
            code_match = re.search(
                r"```(?:\w+)?\s*(.*?)\s*```", text_without_think, re.DOTALL
            )
            if code_match:
                return code_match.group(1).strip()
            else:
                return text_without_think

        df["logic"] = df["extracted_data"].apply(extract_think)
        df["response"] = df["extracted_data"].apply(extract_response)
        return df

    def parse_response(self, response_str: str):
        """
        Извлекает словарь или список словарей из строки 'response'.
        """
        logger.debug("Начало парсинга ответа: %s", response_str)

        cleaned_str = re.sub(r"\.\.\.", "", response_str or "")
        cleaned_str = cleaned_str.strip()

        if not cleaned_str:
            logger.warning("Пустая строка ответа после очистки.")
            return {}

        if cleaned_str.startswith("(") and cleaned_str.endswith(")"):
            cleaned_str = "[" + cleaned_str[1:-1] + "]"
            logger.debug(
                "Обнаружены кортежные скобки. Заменены на квадратные: %s", cleaned_str
            )

        if cleaned_str.startswith("[") and cleaned_str.endswith("]"):
            try:
                result = ast.literal_eval(cleaned_str)
                if isinstance(result, list):
                    logger.debug("Извлечен список словарей через ast.literal_eval.")
                    return result
                else:
                    logger.warning(
                        "Ожидался список словарей, но получен другой тип данных."
                    )
                    return {}
            except Exception as e:
                logger.error("Ошибка при разборе списка словарей: %s", e)
                return {}

        match = re.search(r"(\{.*\})", cleaned_str, re.DOTALL)
        if match:
            dict_str = match.group(1)
            try:
                result = ast.literal_eval(dict_str)
                if isinstance(result, dict):
                    logger.debug("Парсинг словаря через ast.literal_eval успешен.")
                    return result
                else:
                    logger.warning("Ожидался словарь, но получен другой тип данных.")
                    return {}
            except Exception as e:
                logger.error("Ошибка при разборе словаря", e)
                return {}
        else:
            logger.warning("Словарь не найден в строке ответа.")
            return {}

    def fix_response(self, response):
        if isinstance(response, dict):
            return [response]
        if isinstance(response, list):
            return response
        if isinstance(response, tuple):
            return list(response)
        return response

    def safe_geocode_with_tags(self, query: str):
        """
        Безопасное геокодирование запроса через osmnx с получением OSM тегов.
        """
        try:
            gdf = ox.geocode_to_gdf(query)
            if gdf.empty:
                return None
            data = gdf.iloc[0].to_dict()
            return data
        except Exception as e:
            error_message = str(e).lower()
            logger.info(f"OSM error: {error_message}")
            return None

    def replace_nan_in_column(self, series: pd.Series) -> pd.Series:
        """
        Заменяет значения np.nan или списки, содержащие только np.nan, на None.
        """

        def replace_value(x):
            if isinstance(x, list):
                if x and all(pd.isna(item) for item in x):
                    return None
                return x
            if pd.isna(x):
                return None
            return x

        return series.apply(replace_value)

    @staticmethod
    def combine_tags(row):
        """
        Объединяет теги OSM в список строк вида "класс:тип".
        Если встречается вложенный список, его элементы объединяются через запятую.
        """
        if row["osm_class"] is None or row["osm_type"] is None:
            return None

        def process_element(elem):
            if isinstance(elem, list):
                return ",".join(map(str, elem))
            return str(elem)

        processed_class = process_element(row["osm_class"])
        processed_type = process_element(row["osm_type"])

        classes = processed_class.split(",")
        types = processed_type.split(",")

        return [f"{cls}:{typ}" for cls, typ in zip(classes, types)]

    def unique_list(self, x):
        return list(dict.fromkeys(x))

    async def process_texts(self, texts: pd.DataFrame) -> gpd.GeoDataFrame:
        """
        Основной процесс обработки DataFrame texts:
         - отправка запросов,
         - разделение и парсинг извлечённых данных,
         - геокодирование и агрегирование.
        """
        items = [{"context": row["text"]} for _, row in texts.iterrows()]
        descriptions = await self.process_write_descriptions(items)
        texts["extracted_data"] = descriptions
        texts = self.split_extracted_data(texts)
        texts["response"] = texts["response"].apply(self.parse_response)
        texts["response"] = texts["response"].map(self.fix_response)
        texts = texts[texts["response"].map(lambda x: len(x)) > 0]

        named_objects = texts.explode("response")
        named_objects = named_objects[
            ["message_id", "text", "Location", "geometry", "logic", "response"]
        ]
        named_objects["object_name"] = named_objects["response"].map(
            lambda x: x.get("name", None)
        )
        named_objects["object_location"] = named_objects["response"].map(
            lambda x: x.get("location", None)
        )
        named_objects["object_description"] = named_objects["response"].map(
            lambda x: x.get("notes", None)
        )
        named_objects.rename(columns={"geometry": "street_geometry"}, inplace=True)
        named_objects["query"] = (
            named_objects["object_name"] + ", " + named_objects["object_location"]
        )

        named_objects["query_result"] = named_objects["query"].progress_map(
            lambda x: self.safe_geocode_with_tags(x)
        )
        # TODO: починить обращения в OSM. Сейчас либо не находит нужных объектов, либо возникают проблемы с мультиполигонами (?)
        if len(named_objects["query_result"]) == 0:
            logger.info("Data from OSM collected")
            parsed_df = pd.json_normalize(named_objects.query_result)
            parsed_df = parsed_df.drop(
                columns=[
                    "bbox_north",
                    "bbox_south",
                    "bbox_east",
                    "bbox_west",
                    "lat",
                    "lon",
                ],
                errors="ignore",
            )

            parsed_df["geometry"] = (
                parsed_df["geometry"].to_crs(3857).centroid.to_crs(4326)
            )

            named_objects = pd.concat(
                [named_objects.reset_index(drop=True), parsed_df], axis=1
            )
            named_objects["geometry"] = named_objects["geometry"].fillna(
                gpd.GeoSeries.from_wkt(named_objects["street_geometry"])
            )
        else:
            logger.info("No OSM data found")
            named_objects["geometry"] = named_objects["street_geometry"]
            named_objects["class"] = None
            named_objects["type"] = None
            named_objects["type"] = None
            named_objects["osm_id"] = None
            named_objects["display_name"] = None

        named_objects.drop(
            columns=[
                "response",
                "object_location",
                "index",
                "logic",
                "query",
                "place_id",
                "query_result",
                "street_geometry",
                "osm_type",
                "importance",
                "place_rank",
                "addresstype",
                "name",
            ],
            inplace=True,
            errors="ignore",
        )
        named_objects.rename(
            columns={
                "Location": "street_location",
                "class": "osm_class",
                "type": "osm_type",
                "display_name": "osm_name",
                "count": "message_count",
            },
            inplace=True,
        )

        named_objects = gpd.GeoDataFrame(named_objects, geometry="geometry").set_crs(
            4326
        )
        named_objects = named_objects[
            ~named_objects["osm_type"].isin(
                [
                    "administrative",
                    "city",
                    "government",
                    "town",
                    "townhall",
                    "courthouse",
                    "quarter",
                ]
            )
        ]

        logger.info(f"Named objects processed")

        grouped_df = named_objects.groupby(["geometry", "object_name"]).agg(
            self.unique_list
        )
        group_counts = (
            named_objects.groupby(["geometry", "object_name"]).size().rename("count")
        )
        grouped_df = grouped_df.join(group_counts)
        grouped_df = gpd.GeoDataFrame(grouped_df, geometry="geometry").set_crs(4326)
        logger.info(f"Named objects grouped")

        grouped_df["osm_id"] = self.replace_nan_in_column(grouped_df["osm_id"])
        grouped_df["osm_class"] = self.replace_nan_in_column(grouped_df["osm_class"])
        grouped_df["osm_type"] = self.replace_nan_in_column(grouped_df["osm_type"])
        grouped_df["osm_name"] = self.replace_nan_in_column(grouped_df["osm_name"])

        grouped_df["osm_tag"] = grouped_df.apply(self.combine_tags, axis=1)
        grouped_df.drop(
            columns=["osm_class", "osm_type"], inplace=True, errors="ignore"
        )
        grouped_df = grouped_df[
            ~grouped_df.object_name.isin(["Александр Дрозденко", "Игорь Самохин"])
        ]

        return grouped_df

    async def extract_named_objects(self, top: int = None) -> dict:
        """
        1) Ищет все сообщения, у которых is_processed=False (ограничение top, если задан).
        2) Формирует DataFrame (message_id, text, Location, geometry=WKT).
        3) Вызывает self.process_texts(df) -> GeoDataFrame.
        4) Пробегается по результату, создаёт записи в named_object, беря location/geometry из messages.
        5) Коммитит и возвращает информацию о числе добавленных записей.
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
                data.append(
                    {
                        "message_id": msg.message_id,
                        "text": msg.text,
                        "Location": msg.location,
                        "geometry": wkt.dumps(msg.geometry) if msg.geometry else None,
                    }
                )
            df = pd.DataFrame(data)
            result_gdf = await self.process_texts(df)
            if result_gdf.empty:
                return {
                    "detail": "NER extraction returned no named objects.",
                    "processed_messages": len(messages),
                }
            named_objects_created = 0
            for _, row in result_gdf.iterrows():
                mid = row.get("message_id")
                msg = next((m for m in messages if m.message_id == mid), None)
                if not msg:
                    logger.info("No source message for message_id=%s!", mid)
                    continue

                named_obj = NamedObject(
                    text_id=msg.message_id,
                    object_description=row.get("object_description"),
                    osm_id=row.get("osm_id"),
                    osm_tag=row.get("osm_tag"),
                    count=row.get("count", 1),
                    accurate_location=row.get("osm_name"),
                    estimated_location=msg.location,
                    geometry=row.get("geometry"),
                    is_processed=False,
                )
                session.add(named_obj)
                named_objects_created += 1

            await session.commit()

        return {
            "detail": f"Created {named_objects_created} named_object records.",
            "processed_messages": len(messages),
        }

    async def add_named_objects(self, file: UploadFile):
        """
        Обработка CSV-файла для добавления объектов (NamedObject) и восстановления связей в таблице MessageNamedObject.
        """
        content = await file.read()
        decoded = content.decode("utf-8")
        csv_reader = csv.DictReader(StringIO(decoded))
        named_objects = []
        
        async with database.session() as session:
            for row in csv_reader:
                geom_str = row.get("geometry")
                if geom_str:
                    try:
                        if geom_str.strip().upper().startswith("POINT"):
                            point = wkt.loads(geom_str)
                        else:
                            x_str, y_str = geom_str.split(",")
                            point = Point(float(x_str.strip()), float(y_str.strip()))
                    except Exception as e:
                        raise Exception(f"Неверный формат геометрии '{geom_str}'. Ошибка: {e}")
                    ewkt_geometry = preprocessing.to_ewkt(point)
                else:
                    raise Exception("Отсутствует значение геометрии")
                
                object_name = row.get("object_name")
                
                raw_object_description = row.get("object_description")
                try:
                    desc_candidate = ast.literal_eval(raw_object_description)
                    if isinstance(desc_candidate, list):
                        object_description = "; ".join(map(str, desc_candidate))
                    else:
                        object_description = str(desc_candidate)
                except Exception:
                    object_description = raw_object_description
                
                raw_osm_id = row.get("osm_id")
                osm_id = 0
                if raw_osm_id:
                    try:
                        osm_candidate = ast.literal_eval(raw_osm_id)
                        if isinstance(osm_candidate, list):
                            first_osm = osm_candidate[0] if osm_candidate else ""
                        else:
                            first_osm = osm_candidate
                        if first_osm in [None, "", ""]:
                            osm_id = 0
                        else:
                            osm_id = int(float(first_osm))
                    except Exception:
                        try:
                            osm_id = int(float(raw_osm_id))
                        except Exception:
                            osm_id = 0
                
                count = int(row.get("count")) if row.get("count") else None
                osm_tag = row.get("osm_tag")
                
                raw_osm_name = row.get("osm_name")
                if raw_osm_name:
                    try:
                        osm_name_candidate = ast.literal_eval(raw_osm_name)
                        if isinstance(osm_name_candidate, list):
                            accurate_location = str(osm_name_candidate[0]) if osm_name_candidate else ""
                        else:
                            accurate_location = str(osm_name_candidate)
                    except Exception:
                        accurate_location = raw_osm_name
                else:
                    accurate_location = None
                
                street_location = row.get("street_location")
                street_location_list = []
                if street_location:
                    try:
                        street_location_candidate = ast.literal_eval(street_location)
                        if isinstance(street_location_candidate, list):
                            street_location_list = street_location_candidate
                        else:
                            street_location_list = [street_location_candidate]
                    except Exception:
                        street_location_list = [street_location]
                estimated_location = "; ".join(map(str, street_location_list)) if street_location_list else None
                
                text_field = row.get("text")
                text_list = []
                if text_field:
                    try:
                        text_candidate = ast.literal_eval(text_field)
                        if isinstance(text_candidate, list):
                            text_list = text_candidate
                        else:
                            text_list = [text_candidate]
                    except Exception:
                        text_list = [text_field]
                
                linked_message_ids = []
                if text_list:
                    if street_location_list and len(text_list) == len(street_location_list):
                        for t, loc in zip(text_list, street_location_list):
                            result = await session.execute(
                                select(Message).where(
                                    Message.text == t,
                                    Message.location == loc
                                )
                            )
                            msg_obj = result.scalars().first()
                            if msg_obj:
                                linked_message_ids.append(msg_obj.message_id)
                    else:
                        for t in text_list:
                            result = await session.execute(
                                select(Message).where(Message.text == t)
                            )
                            msg_obj = result.scalars().first()
                            if msg_obj:
                                linked_message_ids.append(msg_obj.message_id)
                
                text_id = linked_message_ids[0] if linked_message_ids else None
                
                named_obj = NamedObject(
                    object_name=object_name,
                    estimated_location=estimated_location,
                    object_description=object_description,
                    osm_id=osm_id,
                    accurate_location=accurate_location,
                    count=count,
                    text_id=text_id,
                    osm_tag=osm_tag,
                    geometry=ewkt_geometry,
                    is_processed=True
                )
                session.add(named_obj)
                await session.flush()
                
                for msg_id in linked_message_ids:
                    link = MessageNamedObject(message_id=msg_id, named_object_id=named_obj.named_object_id)
                    session.add(link)
                
                named_objects.append(named_obj)
            await session.commit()
        return named_objects

class IndicatorDefinition:
    def __init__(self):
        self.url = config.get("GPU_URL")
        self.client_cert = config.get("GPU_CLIENT_CERTIFICATE")
        self.client_key = config.get("GPU_CLIENT_KEY")
        self.ca_cert = config.get("GPU_CERTIFICATE")

    def construct_prompt(self, context):
        """
        Формирует промпт для модели на основе текста обращения.
        """
        logger.debug("Начало формирования промпта. Исходный context: %s", context)
        context_str = "\n".join(context) if isinstance(context, list) else str(context)
        prompt = f"""
            Найди обсуждение показателей в тексте {context_str}. 
            Строительство - обращение упоминает строительство новых объектов, реновацию или реконструкцию, открытие объектов, постройку.
            Снос - обращение упоминает уничтожение объектов, их разрушение, повреждение. Любая утрата объекта или его части умышленным образом. 
            Обеспеченность - обращение упоминает то, насколько сервис(объект) загружен жителями. Признаки чрезмерной загруженности - очереди, нехватка мест, нехватка персонала.
            Доступность - обращение упоминает сложности с достижением объекта или сервиса в разумное время. Слишком долго или сложно добираться, слишком большое расстояние до ближайшего объекта. 
            В ответе должен быть список из упомянутых показателей в формате [ind1, ind2, ind3]. Если показатели не обсуждаются, пиши []. Не пиши ничего, кроме списка. Нужен ответ правильного формата.
            """
        logger.debug("Сформированный промпт: %s", prompt)
        return prompt

    async def describe_async(self, context):
        """
        Асинхронно отправляет запрос с сформированным промптом.
        """
        prompt = self.construct_prompt(context)
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "llama3.3",
            "temperature": 0.1,
            "prompt": prompt,
            "stream": False,
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
                        "Ошибка запроса: %s, ответ: %s",
                        response.status_code,
                        response.text,
                    )
                    return None
            except requests.exceptions.RequestException as e:
                logger.error("Ошибка соединения при запросе: %s", e)
                return None

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, sync_request)
        return result

    async def process_find_indicators(self, items):
        """
        Обрабатывает список словарей с ключом 'context' и возвращает список результатов.
        """
        tasks = [self.describe_async(item["context"]) for item in items]
        pbar = tqdm(total=len(tasks), desc="В процессе")

        async def run_task(task):
            result = await task
            pbar.update(1)
            return result

        results = await asyncio.gather(*(run_task(task) for task in tasks))
        pbar.close()
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

    async def get_indicators(self, df):
        """
        Принимает DataFrame с колонкой 'text', обрабатывает запросы и возвращает Series с результатами.
        """
        items = [{"context": row["text"]} for _, row in df.iterrows()]
        indicators = await self.process_find_indicators(items)
        df["indicators"] = indicators
        df["indicators"] = df["indicators"].map(self.parse_indicator_response)
        return df["indicators"].tolist()

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
            df = pd.DataFrame(data)

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


preprocessing = Preprocessing()
indicators_definition = IndicatorDefinition()
ner_extraction = NER_EXTRACTOR()
