import asyncio
import csv
from io import StringIO
from shapely.geometry import Point
from shapely import wkt
from tqdm import tqdm
import osmnx as ox
import logging
from loguru import logger as loglogger

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
