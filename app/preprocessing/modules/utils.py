import asyncio
import csv
from io import StringIO
from shapely.geometry import Point
from shapely import wkt
from tqdm import tqdm
import osmnx as ox
import logging
from loguru import logger as loglogger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.common.db.database import (
    Message,
    MessageStatus,
    Status,
    Group,
    Territory
)

logger = logging.getLogger(__name__)

async def read_csv_to_dict(file) -> list[dict]:
    """
    Асинхронное чтение CSV-файла и возврат списка словарей.
    Блокирующая операция парсинга CSV обёрнута в asyncio.to_thread.
    """
    content = await file.read()
    decoded = content.decode("utf-8")
    
    def parse_csv(decoded_str: str) -> list[dict]:
        reader = csv.DictReader(StringIO(decoded_str))
        return list(reader)
    
    return await asyncio.to_thread(parse_csv, decoded)


def parse_geometry_str(geom_str: str) -> Point:
    """
    Парсит строку геометрии в объект Point.
    Поддерживает форматы "POINT(x y)" и "x, y".
    """
    try:
        if geom_str.strip().upper().startswith("POINT"):
            return wkt.loads(geom_str)
        else:
            x_str, y_str = geom_str.split(",")
            return Point(float(x_str.strip()), float(y_str.strip()))
    except Exception as e:
        raise Exception(f"Неверный формат геометрии '{geom_str}'. Ошибка: {e}")


async def gather_with_progress(tasks: list, description: str = "Processing") -> list:
    """
    Выполняет асинхронное выполнение списка задач с отображением прогресс-бара
    и логированием через logger.info().
    """
    total = len(tasks)
    pbar = tqdm(total=total, desc=description)

    async def run_task(idx: int, task):
        result = await task
        pbar.update(1)
        loglogger.info(f"{description}: task {idx}/{total} completed")
        return result

    wrapped = (run_task(i + 1, task) for i, task in enumerate(tasks))
    results = await asyncio.gather(*wrapped)
    pbar.close()
    return results


async def safe_geocode(query: str) -> dict | None:
    """
    Асинхронное безопасное геокодирование через osmnx.
    Возвращает словарь с данными первой найденной записи или None.
    """
    try:
        gdf = await asyncio.to_thread(ox.geocode_to_gdf, query)
        if gdf.empty:
            return None
        return gdf.iloc[0].to_dict()
    except Exception as e:
        logger.error(f"OSM error for query '{query}': {e}")
        return None


def to_ewkt(geom: Point, srid: int = 4326) -> str:
    """
    Формирует строку вида SRID=4326;POINT(x y)
    """
    return f"SRID={srid};POINT({geom.x} {geom.y})"


async def osm_geocode(query: str, return_osm_id_only: bool = True) -> int | dict | None:
    """
    Асинхронная функция для геокодирования через osmnx.    
    Параметры:
      query: Строка запроса для геокодирования.
      return_osm_id_only: Если True, возвращает только int-значение osm_id.
                          Если False, возвращает полный словарь с OSM-тегами.
    
    Возвращает:
      int, dict или None, если геокодирование не удалось.
    """
    try:
        gdf = await asyncio.to_thread(ox.geocode_to_gdf, query)
        if gdf.empty:
            return None
        if return_osm_id_only:
            return int(gdf.iloc[0]["osm_id"])
        else:
            return gdf.iloc[0].to_dict()
    except Exception as e:
        logger.error(f"Error geocoding '{query}' via osmnx: {e}")
        return None

async def get_unprocessed_texts(
    session: AsyncSession,
    process_type: str,
    top: Optional[int] = None,
    territory_id: Optional[int] = None,
    ):
        query = (
            select(Message)
            .join(MessageStatus, Message.message_id == MessageStatus.message_id)
            .join(Status, Status.process_status_id == MessageStatus.process_status_id)
            .where(
                MessageStatus.process_status.is_(False),     
                Status.process_status_name == process_type,
            )
        )
        if territory_id is not None:
            query = (
                query.join(Group, Group.group_id == Message.group_id)
                .join(Territory, Territory.name == Group.matched_territory)
                .where(Territory.territory_id == territory_id)
            )
        if top and top > 0:
            query = query.limit(top)
        result = await session.execute(query)
        return result.scalars().all()

async def update_message_status(
    session: AsyncSession,
    message_id: int,
    process_type: str):
    process_status_id: Optional[int] = await session.scalar(
        select(Status.process_status_id).where(
            Status.process_status_name == process_type
        )
    )
    if process_status_id is None:
        raise ValueError(f"Status '{process_type}' not found in dictionary")
    ms: Optional[MessageStatus] = await session.scalar(
        select(MessageStatus).where(
            MessageStatus.message_id == message_id,
            MessageStatus.process_status_id == process_status_id,
        )
    )
    if ms is None:
        ms = MessageStatus(
            message_id=message_id,
            process_status_id=process_status_id,
            process_status=True,
        )
        session.add(ms)
    else:
        ms.process_status = True

    return ms
