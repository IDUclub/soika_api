from app.common.db.database import (
    Message,
    NamedObject,
    MessageNamedObject,
)
from app.common.db.db_engine import database
from app.preprocessing.modules import utils
from app.common.exceptions.http_exception_wrapper import http_exception
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
import shapely.wkt
from app.dependencies import config

class NERCalculation:
    def __init__(self, config: Config):
        self.config = config
        self.url = config.get("GPU_URL")
        self.client_cert = config.get("GPU_CLIENT_CERTIFICATE")
        self.client_key = config.get("GPU_CLIENT_KEY")
        self.ca_cert = config.get("GPU_CERTIFICATE")
        self.llm_name = config.get("LLM_NAME")

    def construct_prompt(self, context):
        """
        Формирует промпт для модели на основе контекста.
        """
        logger.debug("Начало формирования промпта. Исходный context: ", context)
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
        logger.debug("Сформированный промпт: ", prompt)
        return prompt

    async def describe_async(self, context):
        """
        Асинхронно отправляет запрос с сформированным промптом.
        """
        prompt = self.construct_prompt(context)
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.llm_name,
            "temperature": 0.1,
            "prompt": prompt,
            "stream": False,
            "think":False
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
                    logger.info("Получен успешный ответ от модели.", response.status_code)
                    response_json = response.json()
                    return response_json.get("response", "")
                else:
                    logger.error("Ошибка запроса, ответ:", response.status_code, response.text)
                    return None
            except requests.exceptions.RequestException as e:
                logger.error("Ошибка соединения при запросе", e)
                return None

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, sync_request)
        return result

    async def process_write_descriptions(self, items):
        """
        Обрабатывает список элементов items асинхронно.
        """
        logger.info("Начало обработки описаний.", len(items))
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
        logger.debug("Начало парсинга ответа: ", response_str)
        cleaned_str = re.sub(r"\.\.\.", "", response_str or "")
        cleaned_str = cleaned_str.strip()
        if not cleaned_str:
            logger.warning("Пустая строка ответа после очистки.")
            return {}

        if cleaned_str.startswith("(") and cleaned_str.endswith(")"):
            cleaned_str = "[" + cleaned_str[1:-1] + "]"
            logger.debug("Обнаружены кортежные скобки. Заменены на квадратные: ", cleaned_str)

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
                logger.error("Ошибка при разборе списка словарей: ", e)
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
                logger.error("Ошибка при разборе словаря: ", e)
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
            logger.info("OSM error: ", error_message)
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
    def safe_to_geometry(val):
        if val is None:
            return None
        if isinstance(val, str):
            return shapely.wkt.loads(val)
        return val

    @staticmethod
    async def process_async(df):
        named_objects = df.explode("response")
        named_objects = named_objects[["message_id", "text", "Location", "geometry", "logic", "response"]]
        named_objects["object_name"] = named_objects["response"].map(lambda x: x.get("name", None))
        named_objects["object_location"] = named_objects["response"].map(lambda x: x.get("location", None))
        named_objects["object_description"] = named_objects["response"].map(lambda x: x.get("notes", None))
        named_objects.rename(columns={"geometry": "street_geometry"}, inplace=True)
        named_objects["query"] = named_objects["object_name"] + ", " + named_objects["object_location"]
        
        queries = named_objects["query"].tolist()
        tasks = [utils.osm_geocode(query, return_osm_id_only=False) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        named_objects["query_result"] = results

        fallback = False

        if len(named_objects["query_result"]) > 0:
            parsed_df = pd.json_normalize(named_objects.query_result)
            if "geometry" in parsed_df.columns:
                logger.info("Data from OSM collected and geometry column is present")
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
                parsed_df["geometry"] = parsed_df["geometry"].apply(ner_calculation.safe_to_geometry)
                named_objects = pd.concat([named_objects.reset_index(drop=True), parsed_df], axis=1)
                named_objects["geometry"] = named_objects["geometry"].fillna(
                    gpd.GeoSeries.from_wkt(named_objects["street_geometry"])
                )
                named_objects.drop(columns=['street_geometry'], inplace=True)
                named_objects.dropna(subset=['geometry'], inplace=True)
                named_objects = gpd.GeoDataFrame(named_objects, geometry='geometry', crs=4326)
                named_objects["geometry"] = (
                    named_objects["geometry"].to_crs(3857).centroid.to_crs(4326)
                )
            else:
                logger.info("Data from OSM collected but geometry column is missing; falling back to default")
                fallback = True
        else:
            logger.info("No OSM data found")
            fallback = True

        if fallback:
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
        
        named_objects = named_objects[~named_objects["osm_type"].isin(
            [
                "administrative",
                "city",
                "government",
                "town",
                "townhall",
                "courthouse",
                "quarter",
                "state"
            ]
        )]
        logger.info("Named objects processed")
        
        grouped_df = named_objects.groupby(["geometry", "object_name"]).agg(ner_calculation.unique_list)
        group_counts = named_objects.groupby(["geometry", "object_name"]).size().rename("count")
        grouped_df = grouped_df.join(group_counts).reset_index()
        
        grouped_df = gpd.GeoDataFrame(grouped_df, geometry="geometry").set_crs(4326)
        logger.info("Named objects grouped")
        
        grouped_df["osm_id"] = ner_calculation.replace_nan_in_column(grouped_df["osm_id"])
        
        grouped_df["osm_class"] = ner_calculation.replace_nan_in_column(grouped_df["osm_class"])
        grouped_df["osm_type"] = ner_calculation.replace_nan_in_column(grouped_df["osm_type"])
        grouped_df["osm_name"] = ner_calculation.replace_nan_in_column(grouped_df["osm_name"])
        grouped_df["osm_tag"] = grouped_df.apply(ner_calculation.combine_tags, axis=1)
        grouped_df["object_description"] = grouped_df["object_description"].map(lambda x: '; '.join(x))
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
        
        if not descriptions or all(d is None for d in descriptions):
            raise http_exception(
                status_code=400,
                msg="Ошибка соединения при запросах",
                input_data=descriptions,
                detail="Проверьте корректность подключения к LLM"
            )
        
        texts["extracted_data"] = descriptions

        texts = await asyncio.to_thread(ner_calculation.split_extracted_data, texts)
        texts["response"] = texts["response"].apply(ner_calculation.parse_response)
        texts["response"] = texts["response"].map(ner_calculation.fix_response)
        texts = texts[texts["response"].map(lambda x: len(x)) > 0]

        grouped_df = await ner_calculation.process_async(texts)
        return grouped_df


    @staticmethod
    def build_data(msgs):
        data = []
        for msg in msgs:
            geom = to_shape(msg.geometry) if msg.geometry else None
            data.append({
                "message_id": msg.message_id,
                "text": msg.text,
                "Location": msg.location,
                "geometry": wkt.dumps(geom) if geom else None,
            })
        return pd.DataFrame(data)
    
    async def extract_named_objects(self, territory_id: int = None, top: int = None) -> dict:
        """
        1) Ищет все сообщения, у которых process_type='named_objects_processed' ещё не True
        (ограничение top, если задано).
        2) Формирует DataFrame (message_id, text, location, geometry=WKT).
        3) Вызывает process_texts(df) → GeoDataFrame с извлечёнными именованными объектами.
        4) Для каждого результата создаёт NamedObject и связи MessageNamedObject.
        5) Ставит/создаёт статус named_objects_processed в MessageStatus.
        6) Коммитит изменения и возвращает число добавленных записей.
        """
        async with database.session() as session:
            messages = await utils.get_unprocessed_texts(
                session,
                process_type="named_objects_processed",
                top=top,
                territory_id=territory_id,
            )
            if not messages:
                return {
                    "detail": "No unprocessed messages found.",
                    "processed_messages": 0,
                }

            df = await asyncio.to_thread(ner_calculation.build_data, messages)
            result_gdf = await ner_calculation.process_texts(df)
            if result_gdf.empty:
                for msg in messages:
                    await utils.update_message_status(
                        session=session,
                        message_id=msg.message_id,
                        process_type="named_objects_processed",
                    )
                await session.commit()
                return {
                    "detail": "NER extraction returned no named objects.",
                    "processed_messages": len(messages),
                }

            named_objects_created = 0
            for _, row in result_gdf.iterrows():
                mid_list = row.get("message_id")
                if not mid_list or not isinstance(mid_list, list):
                    continue

                primary_mid = mid_list[0]
                msg = next((m for m in messages if m.message_id == primary_mid), None)
                if not msg:
                    continue

                object_name          = row.get("object_name")
                object_description   = "; ".join(map(str, row.get("object_description") or [])) or None
                raw_osm_id           = row.get("osm_id")
                first_osm            = (raw_osm_id[0] if isinstance(raw_osm_id, list) else raw_osm_id) or 0
                try:
                    osm_id = int(float(first_osm)) if first_osm not in ("", None) else 0
                except Exception:
                    osm_id = 0
                raw_osm_name         = row.get("osm_name")
                accurate_location    = (raw_osm_name[0] if isinstance(raw_osm_name, list) else raw_osm_name) or None
                raw_osm_tag          = row.get("osm_tag")
                osm_tag              = (raw_osm_tag[0] if isinstance(raw_osm_tag, list) else raw_osm_tag) or None
                count                = row.get("count")
                estimated_location   = msg.location
                text_id              = primary_mid
                geom                 = row.get("geometry")
                ewkt_geometry        = utils.to_ewkt(geom) if geom else None

                if ewkt_geometry is None:
                    continue

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
                    is_processed=False,
                )
                session.add(named_obj)
                await session.flush()  # нужен id для связей

                for msg_id in mid_list:
                    if any(m.message_id == msg_id for m in messages):
                        session.add(
                            MessageNamedObject(
                                message_id=msg_id,
                                named_object_id=named_obj.named_object_id,
                            )
                        )
                named_objects_created += 1

            for msg in messages:
                await utils.update_message_status(
                    session=session,
                    message_id=msg.message_id,
                    process_type="named_objects_processed",
                )

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

    @staticmethod
    async def get_all_named_objects():
        async with database.session() as session:
            result = await session.execute(select(NamedObject))
            named_objects = result.scalars().all()
        list_of_objects = []
        for no in named_objects:
            list_of_objects.append({
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
        response = {"named_objects": list_of_objects}
        return response

    @staticmethod
    async def upload_named_objects_func(file):
        try:
            named_objects = await ner_calculation.add_named_objects(file)
            return {"inserted_count": len(named_objects)}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @staticmethod
    async def extract_named_objects_func(territory_id: int = None, top: int = None):
        result = await ner_calculation.extract_named_objects(territory_id=territory_id, top=top)
        return result

    @staticmethod
    async def delete_all_named_objects_func():
        async with database.session() as session:
            await session.execute(delete(NamedObject))
            await session.commit()
        return {"detail": "All named objects deleted successfully"}


ner_calculation = NERCalculation(config)
    