from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from loguru import logger
from sqlalchemy import select, delete
from geoalchemy2.shape import to_shape
from app.risk_calculation.logic.preprocessing_methods import (
    preprocessing,
    ner_extraction,
    indicators_definition,
)
from app.risk_calculation.dto.vk_requests_dto import VKGroupsRequest, VKTextsRequest
from app.risk_calculation.dto.territory_dto import TerritoryCreate
from app.risk_calculation.dto.emotion_dto import EmotionCreate
from app.risk_calculation.dto.indicator_dto import IndicatorCreate
from app.common.db.database import database
from app.common.db.database import (
    Territory,
    Group,
    Message,
    NamedObject,
    Emotion,
    Indicator,
    MessageIndicator,
    Service,
    MessageService,
)


territories_router = APIRouter()
groups_router = APIRouter()
messages_router = APIRouter()
named_objects_router = APIRouter()
indicators_router = APIRouter()
services_router = APIRouter()


# TODO: убрать логику из роутеров обработки в методы, аналогично индикаторам или NER
@territories_router.get("/get_territories")
async def get_territories():
    """
    Возвращает список всех записей из таблицы territory.
    """
    async with database.session() as session:
        result = await session.execute(select(Territory))
        territories = result.scalars().all()

    territories_list = []
    for t in territories:
        territories_list.append(
            {
                "territory_id": t.territory_id,
                "name": t.name,
                "matched_territory": t.matched_territory,
            }
        )
    return {"territories": territories_list}


@territories_router.post("/add_territory")
async def create_territory(payload: TerritoryCreate):
    """
    POST-метод для создания новой записи в таблице territory.
    Принимает JSON с полями name, matched_territory (необязательное).
    Возвращает созданную запись.
    """
    try:
        new_territory = await preprocessing.create_territory(payload)
        return {
            "territory_id": new_territory.territory_id,
            "name": new_territory.name,
            "matched_territory": new_territory.matched_territory,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@territories_router.delete("/territories")
async def delete_all_territories():
    """
    Удаляет все записи из таблицы territory.
    """
    async with database.session() as session:
        await session.execute(delete(Territory))
        await session.commit()
    return {"detail": "All territories deleted"}


@groups_router.get("/get_groups")
async def get_groups():
    """
    Возвращает список всех записей из таблицы group.
    """
    async with database.session() as session:
        result = await session.execute(select(Group))
        groups = result.scalars().all()

    groups_list = []
    for g in groups:
        groups_list.append(
            {
                "group_id": g.group_id,
                "name": g.name,
                "group_domain": g.group_domain,
                "matched_territory": g.matched_territory,
            }
        )
    return {"groups": groups_list}


@groups_router.post("/collect_vk_groups")
async def collect_vk_groups(data: VKGroupsRequest):
    result = await preprocessing.search_vk_groups(data.territory_id)
    logger.info(f"VK groups for {data.territory_id} collected and saved to database")
    return {
        "status": f"VK groups for id {data.territory_id} {result} collected and saved to database"
    }


@groups_router.delete("/groups")
async def delete_all_groups():
    """
    Удаляет все записи из таблицы group.
    """
    async with database.session() as session:
        await session.execute(delete(Group))
        await session.commit()
    return {"detail": "All groups deleted"}


@messages_router.get("/get_messages")
async def get_messages(
    only_with_location: bool = Query(
        False, description="Возвращать только записи с непустыми geometry и location"
    )
):
    """
    Возвращает список всех записей из таблицы message.
    Поле geometry приводим к строке (WKT), чтобы сериализовать в JSON.

    Параметры:
      - only_with_location (bool): Если True, возвращаем только те записи,
        у которых geometry и location не пусты.
    """
    async with database.session() as session:
        result = await session.execute(select(Message))
        messages = result.scalars().all()

    if only_with_location:
        messages = [
            m for m in messages if m.geometry is not None and m.location is not None
        ]

    messages_list = []
    for m in messages:
        if m.geometry:
            shapely_geom = to_shape(m.geometry)
            wkt_str = shapely_geom.wkt
        else:
            wkt_str = None

        messages_list.append(
            {
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
            }
        )

    return {"messages": messages_list}


@messages_router.post("/add_messages")
async def upload_messages(file: UploadFile = File(...)):
    """
    POST-метод для загрузки сообщений с их атрибутами в базу.
    Принимает CSV файл с колонками, соответствующими полям модели Message:
    text, date, views, likes, reposts, type, parent_message_id, group_id,
    emotion_id, score, geometry, location, is_processed.
    Возвращает количество успешно добавленных сообщений.
    """
    try:
        inserted_messages = await preprocessing.add_messages(file)
        return {"inserted_count": len(inserted_messages)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@messages_router.post("/collect_vk_texts")
async def collect_vk_texts(data: VKTextsRequest):
    result = await preprocessing.parse_VK_texts(
        territory_id=data.territory_id,
        cutoff_date=data.to_date,
        limit=data.limit
    )
    logger.info(f"VK texts for {data.territory_id} collected and saved to DB")
    return {
        "status": f"VK texts for id {data.territory_id} collected and saved to DB. {len(result)} messages total."
    }


@messages_router.post("/add_emotions")
async def create_emotion(payload: EmotionCreate):
    """
    Создаёт новую запись в таблице emotion.
    Принимает JSON-объект с полями:
      - name: название эмоции (строка)
      - emotion_weight: числовой вес эмоции (float)
    Возвращает созданную запись.
    """
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


@messages_router.post("/determine_emotion")
async def determine_emotion_for_unprocessed_messages():
    """
    POST-метод для массовой классификации эмоций.
    1) Находит все сообщения в таблице message, у которых is_processed=False.
    2) Для каждого сообщения вызывает classify_emotion (анализирует text).
    3) Находит соответствующую эмоцию в таблице emotion по полю name.
    4) Присваивает message.emotion_id и выставляет is_processed=True.
    5) Сохраняет изменения в БД.
    Возвращает информацию о количестве обновлённых сообщений.
    """
    async with database.session() as session:
        query = select(Message).where(Message.is_processed == False)
        result = await session.execute(query)
        messages = result.scalars().all()

        if not messages:
            logger.info("No unprocessed messages found.")
            return {"detail": "No unprocessed messages found."}

        total_messages = len(messages)
        logger.info(
            f"Found {total_messages} unprocessed messages. Starting emotion classification..."
        )

        count_updated = 0

        for i, msg in enumerate(messages, start=1):
            label = await preprocessing.classify_emotion(msg.text)

            emotion_query = select(Emotion).where(Emotion.name == label)
            emotion_result = await session.execute(emotion_query)
            emotion_obj = emotion_result.scalar_one_or_none()

            if emotion_obj:
                msg.emotion_id = emotion_obj.emotion_id
                # msg.is_processed = True
                count_updated += 1

        await session.commit()

    logger.info(
        f"Emotion classification done. Updated {count_updated} messages out of {total_messages}."
    )
    return {"detail": f"Processed {count_updated} messages out of {total_messages}."}


@messages_router.post("/extract_addresses")
async def extract_addresses_for_unprocessed_messages(
    device: str = "cpu",
    top: int = Query(None, description="Сколько сообщений обрабатывать (None = все)"),
    territory_name: str = Query("Ленинградская область", description="Название региона для геокодирования")
):
    """
    POST-метод для массового извлечения адресов (location, geometry) из текстов
    в таблице messages, у которых is_processed=False.
    """
    updated_records = await preprocessing.extract_addresses_for_unprocessed(
        device=device,
        top=top,
        input_territory_name=territory_name
    )
    logger.info(
        f"Extraction of addresses completed. Updated {len(updated_records)} messages."
    )
    return {
        "status": f"Extraction of addresses completed. Updated {len(updated_records)} messages."
    }



@messages_router.delete("/messages")
async def delete_all_messages():
    """
    Удаляет все записи из таблицы message.
    """
    async with database.session() as session:
        await session.execute(delete(Message))
        await session.commit()
    return {"detail": "All messages deleted"}


@named_objects_router.get("/get_named_objects")
async def get_named_objects():
    """
    Возвращает список всех записей из таблицы named_object.
    Поле geometry тоже приводим к строке.
    """
    async with database.session() as session:
        result = await session.execute(select(NamedObject))
        named_objs = result.scalars().all()

    named_objects_list = []
    for no in named_objs:
        named_objects_list.append(
            {
                "named_object_id": no.named_object_id,
                "estimated_location": no.estimated_location,
                "object_description": no.object_description,
                "osm_id": no.osm_id,
                "accurate_location": no.accurate_location,
                "count": no.count,
                "text_id": no.text_id,
                "osm_tag": no.osm_tag,
                "geometry": no.geometry.wkt if no.geometry else None,
                "is_processed": no.is_processed,
            }
        )
    return {"named_objects": named_objects_list}


@named_objects_router.post("/extract_named_objects")
async def extract_named_objects_route(
    top: int = Query(
        None, description="Сколько сообщений обрабатывать за один вызов (None = все)"
    )
):
    """
    POST-метод: вызывает функцию extract_named_objects из NER_EXTRACTOR.
    """
    result = await ner_extraction.extract_named_objects(top=top)
    return result


@indicators_router.get("/get_indicators")
async def get_indicators():
    """
    Возвращает список всех записей из таблицы indicators.
    """
    async with database.session() as session:
        result = await session.execute(select(Indicator))
        indicators = result.scalars().all()

    indicators_list = []
    for i in indicators:
        indicators_list.append(
            {
                "indicator_id": i.indicator_id,
                "name": i.name,
            }
        )
    return {"indicators": indicators_list}


@indicators_router.get("/get_message_indicator_pairs")
async def get_message_indicator_pairs():
    """
    Возвращает список всех записей из таблицы message_indicators.
    """
    async with database.session() as session:
        result = await session.execute(select(MessageIndicator))
        indicators = result.scalars().all()

    indicators_list = []
    for i in indicators:
        indicators_list.append(
            {"message_id": i.message_id, "indicator_id": i.indicator_id}
        )
    return {"message_indicator_pairs": indicators_list}


@indicators_router.post("/add_indicators")
async def create_indicator(payload: IndicatorCreate):
    """
    Создаёт новую запись в таблице service.
    Принимает JSON-объект с полями:
      - name: название показателя (строка)
    Возвращает созданную запись.
    """
    async with database.session() as session:
        new_indicator = Indicator(
            name=payload.name,
        )
        session.add(new_indicator)
        await session.commit()
        await session.refresh(new_indicator)

    return {"indicator_id": new_indicator.indicator_id, "name": new_indicator.name}


@indicators_router.post("/extract_indicators")
async def extract_indicators_route(
    top: int = Query(
        None, description="Сколько сообщений обрабатывать за один вызов (None = все)"
    )
):
    """
    POST-метод: проходит по всем is_processed=False сообщениям, находит индикаторы
    и складывает их в message_indicator.
    """
    result = await indicators_definition.extract_indicators(top=top)
    return result


@indicators_router.delete("/indicators")
async def delete_all_indicators():
    """
    Удаляет все записи из таблицы indicators.
    """
    async with database.session() as session:
        await session.execute(delete(Indicator))
        await session.commit()
    return {"detail": "All indicators deleted"}


@services_router.get("/get_services")
async def get_services():
    """
    Возвращает список всех записей из таблицы services.
    """
    async with database.session() as session:
        result = await session.execute(select(Service))
        services = result.scalars().all()

    services_list = []
    for s in services:
        services_list.append(
            {"service_id": s.service_id, "name": s.name, "value_id": s.value_id}
        )
    return {"services": services_list}


@services_router.get("/get_message_service_pairs")
async def get_message_service_pairs():
    """
    Возвращает список всех записей из таблицы message_services.
    """
    async with database.session() as session:
        result = await session.execute(select(MessageService))
        services = result.scalars().all()

    services_list = []
    for s in services:
        services_list.append({"message_id": s.message_id, "indicator_id": s.service_id})
    return {"message_service_pairs": services_list}


@services_router.post("/extract_services")
async def extract_services_route(
    top: int = Query(
        None, description="Сколько сообщений обрабатывать за один вызов (None = все)"
    )
):
    """
    POST-метод: проходит по всем is_processed=False сообщениям,
    выявляет сервисы, складывает в message_service.
    """
    result = await preprocessing.extract_services_in_messages(top=top)
    return result
