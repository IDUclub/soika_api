#  TODO: зарефакторить этот модуль

from app.common.db.database import (
    Territory,
    Group,
    Message,
    Emotion,
    Territory,
    GroupTerritory,
    Indicator,
    MessageIndicator,
    Service,
    MessageService,
)
from app.common.db.db_engine import database
from sqlalchemy import select, delete, func
from geoalchemy2.shape import to_shape
import asyncio
import numpy as np
from loguru import logger
import pandas as pd
from soika import Geocoder, VKParser
from app.common.modules.constants import CONSTANTS
from shapely.geometry import Point
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from fastapi import UploadFile, HTTPException
from app.preprocessing.modules import utils
from app.preprocessing.modules.models import models_initialization
from app.preprocessing.modules.groups import groups_calculation
from app.preprocessing.modules.services import services_calculation
import concurrent.futures
from iduconfig import Config
from app.dependencies import config

class MessagesCalculation:
    def __init__(self, config: Config):
        self.config = config

    async def add_messages(self, file: UploadFile):
        """
        Обработка CSV-файла для добавления сообщений в базу данных.
        """
        logger.info(f"Начало обработки CSV-файла '{file.filename}' для добавления сообщений.")

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

        records = await asyncio.to_thread(utils.read_csv_to_dict, file)
        messages = []
        ru_service_names = CONSTANTS.json["ru_service_names"]

        async with database.session() as session:
            for row in records:
                result = await session.execute(select(Emotion).where(Emotion.name == row.get("emotion")))
                emotion_obj = result.scalars().first()
                if not emotion_obj:
                    logger.error(f"Эмоция '{row.get('emotion')}' не найдена в базе данных")
                    raise Exception(f"Эмоция '{row.get('emotion')}' не найдена в базе данных")
                emotion_id = emotion_obj.emotion_id

                geom_str = row.get("geometry")
                if geom_str:
                    point = utils.parse_geometry_str(geom_str)
                    ewkt_geometry = utils.to_ewkt(point)
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
                    services_list = services_calculation.replace_service_names(services_list, ru_service_names)
                    services_list = list(dict.fromkeys(services_list))
                    with session.no_autoflush:
                        for service_name in services_list:
                            result = await session.execute(select(Service).where(Service.name == service_name))
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
                            result = await session.execute(select(Indicator).where(Indicator.name == indicator_name))
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
        logger.info(f"Обработка файла '{file.filename}' завершена. Добавлено сообщений: {len(messages)}")
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
            latest_date = await messages_calculation.get_latest_message_date_by_territory_id(territory_id)
            if latest_date:
                next_day = latest_date + timedelta(days=1)
                cutoff_date = next_day.strftime("%Y-%m-%d")
            else:
                two_years_ago = datetime.now() - relativedelta(years=2)
                cutoff_date = two_years_ago.strftime("%Y-%m-%d")

        access_key = config.get("VK_ACCESS_KEY")
        parser = VKParser()
        groups = await groups_calculation.get_groups_by_territory_id(territory_id)
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

            logger.info(f"Processing group_id={group.group_id}, domain={group.group_domain}...")

            df = await asyncio.to_thread(
                parser.run_parser,
                domain=group.group_domain,
                access_token=access_key,
                cutoff_date=cutoff_date,
            )

            if df.empty:
                logger.info(f"No new messages for group_id={group.group_id}. Skipping.")
                continue

            if limit is not None:
                remaining = limit - total_messages_count
                df = await asyncio.to_thread(lambda d: d.head(remaining), df)

            df["group_id"] = group.group_id
            df = await asyncio.to_thread(lambda d: d.replace({np.nan: None}), df)
            df["date"] = await asyncio.to_thread(lambda d: pd.to_datetime(d["date"], utc=True), df)
            messages_data = await asyncio.to_thread(lambda d: d.to_dict("records"), df)

            for message in messages_data:
                message.pop("id", None)  # Удаляем оригинальный vk id
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

            logger.info(f"Committed {len(messages_data)} messages to DB for group_id={group.group_id}.")
            total_messages_count += len(messages_data)
            all_messages_data.extend(messages_data)

        logger.info(f"Finished parsing territory_id={territory_id}. Total messages processed: {total_messages_count}.")
        return all_messages_data

    async def classify_emotion(self, text: str):
        """
        Асинхронная функция для получения предсказания по тексту с помощью модели классификации.
        """
        model = models_initialization._classification_model
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(models_initialization.executor, model, text)
        return result[0]["label"]

    @staticmethod
    def run_geocoding_task(osm_key, msgs, device, territory_name):
        df = pd.DataFrame({
            "message_id": [m.message_id for m in msgs],
            "text": [m.text for m in msgs]
        })
        if df.empty:
            return None
        geocoder = Geocoder(
            df=df,
            osm_id=osm_key,
            device=device,
            model_path="Geor111y/flair-ner-addresses-extractor",
            text_column_name="text",
            city_tags={"admin_level": ["6"]},
            territory_name=territory_name,
            nb_workers=-1
        )
        logger.info(f"Started geocoding process for messages in territory '{territory_name}' (osm_id={osm_key})")
        result_gdf = geocoder.run(group_column=None, search_for_objects=False)
        if "message_id" not in result_gdf.columns:
            logger.warning(f"Geocoder result has no 'message_id' column. Skipping osm_id={osm_key}.")
            return None
        return result_gdf.to_dict("records")

    async def extract_addresses_for_unprocessed(self, device: str = "cpu", top: int = None, input_territory_name: str = None) -> list[dict]:
        """
        1) Получает все сообщения (Message) с is_processed=False (ограничиваем top, если задан).
        2) Получает группы (Group) для этих сообщений.
        3) Если input_territory_name задан, обрабатывает все сообщения как относящиеся к этой территории.
        4) Если input_territory_name не задан, получает уникальные территории из групп и для каждой вызывает utils.osm_geocode для получения osm_id.
        5) Группирует сообщения по osm_id, пакетно вызывает geocoder.run() в отдельном процессе.
        6) Обновляет поля location, geometry, is_processed.
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

            if input_territory_name:
                osm_id = await utils.osm_geocode(input_territory_name)
                if not osm_id:
                    logger.error(f"Could not determine osm_id for territory='{input_territory_name}'.")
                    return []
                messages_by_osm = {osm_id: messages}
            else:
                group_ids = {m.group_id for m in messages if m.group_id is not None}
                if not group_ids:
                    logger.info("No valid group_id found among unprocessed messages.")
                    return []

                grp_query = select(Group).where(Group.group_id.in_(group_ids))
                grp_result = await session.execute(grp_query)
                groups = grp_result.scalars().all()
                group_map = {g.group_id: g for g in groups}

                territory_names = {g.matched_territory for g in groups if g.matched_territory}
                if not territory_names:
                    logger.info("No territory names found in groups.")
                    return []

                territory_osm_ids = {}
                for name in territory_names:
                    osm_id = await utils.osm_geocode(name)
                    if osm_id:
                        territory_osm_ids[name] = osm_id
                        logger.info(f"OSM id for territory '{name}': {osm_id}")
                    else:
                        logger.warning(f"Could not determine osm_id for territory '{name}'.")

                messages_by_osm = {}
                for msg in messages:
                    grp = group_map.get(msg.group_id)
                    if not grp:
                        logger.warning(f"Message {msg.message_id} has invalid group_id={msg.group_id}. Skipping.")
                        continue
                    territory_name = grp.matched_territory
                    if not territory_name:
                        logger.warning(f"Group {grp.group_id} has no matched_territory. Skipping message {msg.message_id}.")
                        continue
                    osm_id = territory_osm_ids.get(territory_name)
                    if not osm_id:
                        logger.warning(f"Could not determine osm_id for territory '{territory_name}'. Skipping msg={msg.message_id}.")
                        continue
                    messages_by_osm.setdefault(osm_id, []).append(msg)

            updated_records = []

            loop = asyncio.get_running_loop()
            with concurrent.futures.ProcessPoolExecutor() as executor:
                tasks = {
                    osm_key: loop.run_in_executor(executor, messages_calculation.run_geocoding_task, osm_key, msgs, device, input_territory_name or None)
                    for osm_key, msgs in messages_by_osm.items()
                }
                for osm_key, future in tasks.items():
                    records = await future
                    if not records:
                        continue
                    msg_map = {m.message_id: m for m in messages_by_osm[osm_key]}
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
                            msg_obj.geometry = utils.to_ewkt(geom_data, srid=4326)
                        else:
                            msg_obj.geometry = None
                        msg_obj.is_processed = True
                        updated_records.append({
                            "message_id": mid,
                            "osm_id": osm_key,
                            "location": str(loc),
                            "geometry": msg_obj.geometry,
                        })
            await session.commit()
            logger.info(f"Batch extraction done. Updated {len(updated_records)} messages.")
            return updated_records

    @staticmethod
    async def get_all_messages(only_with_location: bool = False):
        async with database.session() as session:
            result = await session.execute(select(Message))
            messages = result.scalars().all()
        if only_with_location:
            messages = [m for m in messages if m.geometry is not None and m.location is not None]
        messages_list = []
        for m in messages:
            if m.geometry:
                shapely_geom = to_shape(m.geometry)
                wkt_str = shapely_geom.wkt
            else:
                wkt_str = None
            messages_list.append({
                "message_id": m.message_id,
                "text": m.text,
                "date": m.date.isoformat() if m.date else None,
                "views": m.views,
                "likes": m.likes,
                "reposts": m.reposts,
                "type": m.type,
                "parent_message_id": m.parent_message_id,
                "group_id": m.group_id,
                "emotion_id": m.emotion_id,
                "score": m.score,
                "geometry": wkt_str,
                "location": m.location,
                "is_processed": m.is_processed,
            })
        return {"messages": messages_list}

    @staticmethod
    async def upload_messages_func(file):
        try:
            inserted_messages = await messages_calculation.add_messages(file)
            return {"inserted_count": len(inserted_messages)}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @staticmethod
    async def collect_vk_texts_func(data):
        result = await messages_calculation.parse_VK_texts(
            territory_id=data.territory_id,
            cutoff_date=data.to_date,
            limit=data.limit
        )
        return {"status": f"VK texts for id {data.territory_id} collected and saved to DB. {len(result)} messages total."}

    @staticmethod
    async def create_emotion_func(payload):
        async with database.session() as session:
            new_emotion = Emotion(name=payload.name, emotion_weight=payload.emotion_weight)
            session.add(new_emotion)
            await session.commit()
            await session.refresh(new_emotion)
        return {
            "emotion_id": new_emotion.emotion_id,
            "name": new_emotion.name,
            "emotion_weight": new_emotion.emotion_weight,
        }

    @staticmethod
    async def determine_emotion_for_unprocessed_messages_func():
        async with database.session() as session:
            query = select(Message).where(Message.is_processed == False)
            result = await session.execute(query)
            messages = result.scalars().all()
            if not messages:
                return {"detail": "No unprocessed messages found."}
            total_messages = len(messages)
            count_updated = 0
            for msg in messages:
                label = await messages_calculation.classify_emotion(msg.text)
                emotion_query = select(Emotion).where(Emotion.name == label)
                emotion_result = await session.execute(emotion_query)
                emotion_obj = emotion_result.scalar_one_or_none()
                if emotion_obj:
                    msg.emotion_id = emotion_obj.emotion_id
                    count_updated += 1
            await session.commit()
        return {"detail": f"Processed {count_updated} messages out of {total_messages}."}

    @staticmethod
    async def extract_addresses_for_unprocessed_messages_func(device: str = "cpu", top: int = None, input_territory_name: str = "Ленинградская область"):
        updated_records = await messages_calculation.extract_addresses_for_unprocessed(
            device=device,
            top=top,
            input_territory_name=input_territory_name
        )
        return {"status": f"Extraction of addresses completed. Updated {len(updated_records)} messages."}

    @staticmethod
    async def delete_all_messages_func():
        async with database.session() as session:
            await session.execute(delete(Message))
            await session.commit()
        return {"detail": "All messages deleted"}


messages_calculation = MessagesCalculation(config)