#  TODO: зарефакторить этот модуль

from fastapi import Request

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
from soika import VKParser
from app.common.modules.constants import CONSTANTS
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from fastapi import UploadFile, HTTPException
from app.preprocessing.modules import utils
from app.preprocessing.modules.models import models_initialization
from app.preprocessing.modules.groups import groups_calculation
from app.preprocessing.modules.services import services_calculation
from iduconfig import Config
from app.dependencies import config


class MessagesCalculation:
    def __init__(self, config: Config):
        self.config = config

    async def add_messages(self, file: UploadFile):
        """
        Обработка CSV-файла для добавления сообщений в базу данных.

        Алгоритм работы:
        1. Считываем записи из CSV и проходим по каждой строке.
        2. Для каждой строки:
            - Определяем поля (эмоция, геометрия, территория, группа и т.д.) и создаём объект Message.
            - Устанавливаем parent_message_id = None.
            - Сохраняем объект, чтобы получить новый message_id.
            - Если в CSV присутствует оригинальный id, сохраняем сопоставление CSV_id -> объект Message.
            - Если в CSV указано значение parent_message_id, сохраняем пару (CSV_parent_id, текущий объект Message) для последующей установки связи.
            - Обрабатываем дополнительные связи: сервисы, индикаторы, связь группы с территорией.
        3. После добавления всех сообщений проходим по сохранённым парам для установки parent_message_id,
            ищем в словаре соответствий родительское сообщение по CSV_parent_id и обновляем поле.
        4. Коммитим все изменения.
        """
        logger.info(f"Начало обработки CSV-файла '{file.filename}' для добавления сообщений.")

        records = await utils.read_csv_to_dict(file)
        messages = []
        ru_service_names = CONSTANTS.json["ru_service_names"]

        csv_to_db_message = {}
        parent_relationships = []

        async with database.session() as session:
            for row in records:
                csv_message_id = row.get("id")
                csv_parent_id = row.get("parent_message_id")

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
                        group_obj = Group(name=group_name, group_domain=group_domain,
                                          matched_territory=matched_territory)
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
                    parent_message_id=None,
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

                if csv_message_id:
                    csv_to_db_message[csv_message_id] = message

                if csv_parent_id:
                    parent_relationships.append((csv_parent_id, message))

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

            for csv_parent_id, message in parent_relationships:
                parent_message = csv_to_db_message.get(csv_parent_id)
                if parent_message:
                    message.parent_message_id = parent_message.message_id
                    logger.info(
                        f"Установлена связь: сообщение {message.message_id} с родительским {parent_message.message_id}")
                else:
                    logger.warning(
                        f"Родительское сообщение с CSV id {csv_parent_id} не найдено. Оставляем parent_message_id как None.")
                await session.flush()

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

    async def parse_VK_texts(self, territory_id: int, cutoff_date: str = None, limit: int = None,
                             request: Request = None):
        """
        Получает сообщения из ВК для групп, связанных с territory_id.
        Если параметр request передан, то перед обработкой каждой группы проверяется,
        не был ли прерван запрос клиентом.

        Обновлено для корректного формирования связей между сообщениями:
        - Сначала создаются сообщения с parent_message_id=None.
        - Сохраняется соответствие оригинальных id из VK с новыми message_id.
        - Затем устанавливаются связи между сообщениями на основании оригинальных parent_message_id.
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
            f"Starting to parse VK texts for territory_id={territory_id}, cutoff_date={cutoff_date}, found {len(groups)} groups."
        )

        for group in groups:
            # TODO: убрать после перехода на брокер
            if request is not None and await request.is_disconnected():
                logger.info("Client disconnected. Cancelling further processing.")
                break

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

            orig_to_new = {}
            pending_parent_updates = []

            for message in messages_data:
                orig_id = message.get("id")
                orig_parent_id = message.get("parent_message_id")
                message.pop("id", None)
                message.pop("parents_stack", None)
                message["views"] = message.pop("views.count")
                message["likes"] = message.pop("likes.count")
                message["reposts"] = message.pop("reposts.count")
                message["parent_message_id"] = None
                message["score"] = None
                message["location"] = None
                message["geometry"] = None
                message["emotion_id"] = None
                message["is_processed"] = False

                message["_orig_id"] = orig_id
                message["_orig_parent_id"] = orig_parent_id

            async with database.session() as session:
                for msg in messages_data:
                    new_message = Message(
                        text=msg["text"],
                        date=msg["date"],
                        views=msg["views"],
                        likes=msg["likes"],
                        reposts=msg["reposts"],
                        type=msg["type"],
                        parent_message_id=None,
                        group_id=msg["group_id"],
                        emotion_id=msg["emotion_id"],
                        score=msg["score"],
                        geometry=msg["geometry"],
                        location=msg["location"],
                        is_processed=msg["is_processed"],
                    )
                    session.add(new_message)
                    await session.flush()

                    orig_id = msg.get("_orig_id")
                    if orig_id is not None:
                        orig_to_new[orig_id] = new_message

                    orig_parent_id = msg.get("_orig_parent_id")
                    if orig_parent_id is not None:
                        pending_parent_updates.append((orig_parent_id, new_message))

                for orig_parent, child_message in pending_parent_updates:
                    parent_message = orig_to_new.get(orig_parent)
                    if parent_message:
                        child_message.parent_message_id = parent_message.message_id
                        logger.info(f"Set parent for message {child_message.message_id} to {parent_message.message_id}")
                    else:
                        logger.warning(
                            f"Parent with original id {orig_parent} not found. Leaving parent_message_id as None.")
                    await session.flush()

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
    async def collect_vk_texts_func(data, request: Request):
        result = await messages_calculation.parse_VK_texts(
            territory_id=data.territory_id,
            cutoff_date=data.to_date,
            limit=data.limit,
            request=request
        )
        return {
            "status": f"VK texts for id {data.territory_id} collected and saved to DB. {len(result)} messages total."}

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
    async def delete_all_messages_func():
        async with database.session() as session:
            await session.execute(delete(Message))
            await session.commit()
        return {"detail": "All messages deleted"}


messages_calculation = MessagesCalculation(config)