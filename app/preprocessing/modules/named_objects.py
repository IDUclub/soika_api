from app.common.db.database import (
    Message,
    NamedObject,
    MessageNamedObject,
)
from app.common.db.db_engine import database
from app.preprocessing.modules import utils
from sqlalchemy import select, delete
from geoalchemy2.shape import to_shape
import asyncio
import requests
from loguru import logger
import pandas as pd
import geopandas as gpd
import osmnx as ox
import ast
from shapely import wkt
from fastapi import UploadFile, HTTPException
import csv
from io import StringIO
import re
from iduconfig import Config

config = Config()

class NERCalculation:
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
            logger.info("Отправка запроса на %s", self.url)
            try:
                response = requests.post(
                    self.url,
                    headers=headers,
                    json=data,
                    cert=(self.client_cert, self.client_key),
                    verify=self.ca_cert,
                )
                if response.status_code == 200:
                    logger.info("Получен успешный ответ от модели. Код: %s", response.status_code)
                    response_json = response.json()
                    return response_json.get("response", "")
                else:
                    logger.error("Ошибка запроса: %s, ответ: %s", response.status_code, response.text)
                    return None
            except requests.exceptions.RequestException as e:
                logger.error("Ошибка соединения при запросе: %s", e)
                return None

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, sync_request)
        return result

    async def process_write_descriptions(self, items):
        """
        Обрабатывает список элементов items (каждый со своим 'context') асинхронно.
        """
        logger.info("Начало обработки описаний. Всего элементов: %s", len(items))
        tasks = [self.describe_async(item["context"]) for item in items]
        results = await utils.gather_with_progress(tasks, description="В процессе")
        logger.info("Завершена обработка описаний.")
        return results

    @staticmethod
    def extract_think(text):
            match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
            return match.group(1).strip() if match else ""
    
    @staticmethod
    def extract_response(text):
        text_without_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        code_match = re.search(r"```(?:\w+)?\s*(.*?)\s*```", text_without_think, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        else:
            return text_without_think
    
    def split_extracted_data(self, df):
        """
        Разбивает содержимое столбца 'extracted_data' на два столбца: 'logic' и 'response'.
        """
        

        df["logic"] = df["extracted_data"].apply(ner_calculation.extract_think)
        df["response"] = df["extracted_data"].apply(ner_calculation.extract_response)
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
            logger.debug("Обнаружены кортежные скобки. Заменены на квадратные: %s", cleaned_str)

        if cleaned_str.startswith("[") and cleaned_str.endswith("]"):
            try:
                result = ast.literal_eval(cleaned_str)
                if isinstance(result, list):
                    logger.debug("Извлечен список словарей через ast.literal_eval.")
                    return result
                else:
                    logger.warning("Ожидался список словарей, но получен другой тип данных.")
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
                logger.error("Ошибка при разборе словаря: %s", e)
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
            logger.info("OSM error: %s", error_message)
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

    @staticmethod
    def build_items(df):
        return [{"context": row["text"]} for _, row in df.iterrows()]

    @staticmethod
    def process_sync(df):
            named_objects = df.explode("response")
            named_objects = named_objects[["message_id", "text", "Location", "geometry", "logic", "response"]]
            named_objects["object_name"] = named_objects["response"].map(lambda x: x.get("name", None))
            named_objects["object_location"] = named_objects["response"].map(lambda x: x.get("location", None))
            named_objects["object_description"] = named_objects["response"].map(lambda x: x.get("notes", None))
            named_objects.rename(columns={"geometry": "street_geometry"}, inplace=True)
            named_objects["query"] = named_objects["object_name"] + ", " + named_objects["object_location"]
            named_objects["query_result"] = named_objects["query"].progress_map(
                lambda x: utils.osm_geocode(x, return_osm_id_only=False)
            )
            if len(named_objects["query_result"]) > 0:
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
                named_objects = pd.concat([named_objects.reset_index(drop=True), parsed_df], axis=1)
                named_objects["geometry"] = named_objects["geometry"].fillna(
                    gpd.GeoSeries.from_wkt(named_objects["street_geometry"])
                )
            else:
                logger.info("No OSM data found")
                named_objects["geometry"] = named_objects["street_geometry"]
                named_objects["class"] = None
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
            named_objects = gpd.GeoDataFrame(named_objects, geometry="geometry").set_crs(4326)
            named_objects = named_objects[~named_objects["osm_type"].isin(
                [
                    "administrative",
                    "city",
                    "government",
                    "town",
                    "townhall",
                    "courthouse",
                    "quarter",
                ]
            )]
            logger.info("Named objects processed")
            grouped_df = named_objects.groupby(["geometry", "object_name"]).agg(ner_calculation.unique_list)
            group_counts = named_objects.groupby(["geometry", "object_name"]).size().rename("count")
            grouped_df = grouped_df.join(group_counts)
            grouped_df = gpd.GeoDataFrame(grouped_df, geometry="geometry").set_crs(4326)
            logger.info("Named objects grouped")
            grouped_df["osm_id"] = ner_calculation.replace_nan_in_column(grouped_df["osm_id"])
            grouped_df["osm_class"] = ner_calculation.replace_nan_in_column(grouped_df["osm_class"])
            grouped_df["osm_type"] = ner_calculation.replace_nan_in_column(grouped_df["osm_type"])
            grouped_df["osm_name"] = ner_calculation.replace_nan_in_column(grouped_df["osm_name"])
            grouped_df["osm_tag"] = grouped_df.apply(ner_calculation.combine_tags, axis=1)
            grouped_df.drop(columns=["osm_class", "osm_type"], inplace=True, errors="ignore")
            grouped_df = grouped_df[~grouped_df.object_name.isin(["Александр Дрозденко", "Игорь Самохин"])]
            return grouped_df
    
    async def process_texts(self, texts: pd.DataFrame) -> gpd.GeoDataFrame:
        """
        Основной процесс обработки DataFrame texts:
         - отправка запросов,
         - разделение и парсинг извлечённых данных,
         - геокодирование и агрегирование.
        """
        items = await asyncio.to_thread(ner_calculation.build_items, texts)
        descriptions = await ner_calculation.process_write_descriptions(items)
        texts["extracted_data"] = descriptions

        texts = await asyncio.to_thread(ner_calculation.split_extracted_data, texts)
        texts["response"] = texts["response"].apply(ner_calculation.parse_response)
        texts["response"] = texts["response"].map(ner_calculation.fix_response)
        texts = texts[texts["response"].map(lambda x: len(x)) > 0]

        grouped_df = await asyncio.to_thread(ner_calculation.process_sync, texts)
        return grouped_df

    @staticmethod
    def build_data(msgs):
        data = []
        for msg in msgs:
            data.append({
                "message_id": msg.message_id,
                "text": msg.text,
                "Location": msg.location,
                "geometry": wkt.dumps(msg.geometry) if msg.geometry else None,
            })
        return pd.DataFrame(data)
    
    async def extract_named_objects(self, top: int = None) -> dict:
        """
        1) Ищет все сообщения, у которых is_processed=False (ограничение top, если задан).
        2) Формирует DataFrame (message_id, text, Location, geometry=WKT).
        3) Вызывает self.process_texts(df) -> GeoDataFrame.
        4) Пробегается по результату, создаёт записи в named_object.
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
            df = await asyncio.to_thread(ner_calculation.build_data, messages)
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

    @staticmethod
    def parse_csv(data):
            return list(csv.DictReader(StringIO(data)))
    
    async def add_named_objects(self, file: UploadFile):
        """
        Обработка CSV-файла для добавления объектов (NamedObject) и восстановления связей в таблице MessageNamedObject.
        """
        content = await file.read()
        decoded = content.decode("utf-8")
        
        csv_rows = await asyncio.to_thread(ner_calculation.parse_csv, decoded)
        named_objects = []
        async with database.session() as session:
            for row in csv_rows:
                geom_str = row.get("geometry")
                if geom_str:
                    point = utils.parse_geometry_str(geom_str)
                    ewkt_geometry = utils.to_ewkt(point)
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

    async def get_all_named_objects():
        async with database.session() as session:
            result = await session.execute(select(NamedObject))
            named_objects = result.scalars().all()
        response = []
        for no in named_objects:
            response.append({
                "named_object_id": no.named_object_id,
                "object_name": no.object_name,
                "object_description": no.object_description,
                "estimated_location": no.estimated_location,
                "accurate_location": no.accurate_location,
                "osm_id": no.osm_id,
                "count": no.count,
                "osm_tag": no.osm_tag,
                "text_id": no.text_id,
                "geometry": to_shape(no.geometry).wkt if no.geometry else None,
                "is_processed": no.is_processed,
            })
        return response

    async def upload_named_objects_func(file):
        try:
            named_objects = await ner_calculation.add_named_objects(file)
            return {"inserted_count": len(named_objects)}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def extract_named_objects_func(top: int = None):
        result = await ner_calculation.extract_named_objects(top=top)
        return result

    async def delete_all_named_objects_func():
        async with database.session() as session:
            await session.execute(delete(NamedObject))
            await session.commit()
        return {"detail": "All named objects deleted successfully"}


ner_calculation = NERCalculation()
    