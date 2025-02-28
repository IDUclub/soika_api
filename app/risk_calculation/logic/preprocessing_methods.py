from app.common.config import config
import asyncio
import numpy as np
import requests
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
import pandas as pd
import geopandas as gpd
from sloyka import Geocoder, VKParser
import json
import torch
from flair.models import SequenceTagger
from flair.data import Sentence
from app.risk_calculation.logic.constants import CONSTANTS
import re
from rapidfuzz import fuzz

import logging
import requests
import json
import re
import asyncio
import geopandas as gpd
import pandas as pd
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm
import osmnx as ox
import ast
from shapely.geometry import Point
import numpy as np

class Preprocessing:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._classification_model = None
        self._ner_model = None

    @staticmethod
    def _load_flair_model_cpu(model_name: str) -> SequenceTagger:
        """
        Загружает модель Flair SequenceTagger, принудительно используя устройство CPU.
        Переопределяет torch.load, чтобы параметр map_location всегда был torch.device("cpu").
        """
        orig_torch_load = torch.load

        def torch_load_cpu(*args, **kwargs):
            kwargs["map_location"] = torch.device("cpu")
            return orig_torch_load(*args, **kwargs)

        torch.load = torch_load_cpu
        tagger = Preprocessing._original_sequence_tagger_load(model_name)
        torch.load = orig_torch_load
        return tagger

    # Сохраняем оригинальный метод load для дальнейшего использования внутри _load_flair_model_cpu
    _original_sequence_tagger_load = SequenceTagger.load
    # Переопределяем глобально SequenceTagger.load, чтобы все вызовы шли через наш метод
    SequenceTagger.load = _load_flair_model_cpu.__func__

    async def init_models(self):
        """
        Асинхронная инициализация двух моделей:
        - Модель для классификации эмоций через Transformers pipeline.
        - Модель для извлечения адресов через Flair SequenceTagger (загружается на CPU).
        """
        loop = asyncio.get_event_loop()
        classification_pipeline = "text-classification"
        classification_model_name = "Sandrro/emotions_classificator_v4"
        ner_model_name = "Geor111y/flair-ner-addresses-extractor"

        logger.info(f"Launching classification model {classification_model_name} for {classification_pipeline}")
        self._classification_model = await loop.run_in_executor(
            self.executor,
            lambda: pipeline(classification_pipeline, model=classification_model_name)
        )
        logger.info(f"Launching NER model {ner_model_name} with Flair SequenceTagger (forcing CPU)")
        self._ner_model = await loop.run_in_executor(
            self.executor,
            lambda: SequenceTagger.load(ner_model_name)
        )

    def get_classification_model(self):
        return self._classification_model

    def get_ner_model(self):
        return self._ner_model

    def search_vk_groups(self, territory_name: str, sort: int = 4, count: int = 20, version: str = "5.131") -> pd.DataFrame:
        group_access_key = config.get("VK_GROUP_ACCESS_KEY")
        params = {
            "q": territory_name,
            "sort": sort,
            "count": count,
            "access_token": group_access_key,
            "v": version
        }
        
        response = requests.get("https://api.vk.com/method/groups.search", params=params)
        data = response.json()
        df = pd.DataFrame(data['response']['items'])[['id', 'name', 'screen_name']]
        df.rename(columns={'screen_name': 'domain'}, inplace=True)
        
        return df.to_dict()

    def parse_VK_texts(self, group_domains: str, cutoff_date: str):
        access_key = config.get("VK_ACCESS_KEY")
        parser = VKParser()
        group_domains = group_domains.split(', ')
        result = pd.concat([
            parser.run_parser(
            domain=domain, 
            access_token=access_key,
            cutoff_date=cutoff_date
            ) 
            for domain in group_domains])
        result = result.replace({np.nan: None})
        return result.to_dict()

    async def classify_emotion(self, text: str):
        """
        Асинхронная функция для получения предсказания по тексту с помощью модели классификации.
        """
        model = self.get_classification_model()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, model, text)
        return result[0]['label']

    def extract_addresses_sync(self, text: str):
        """
        Синхронная функция для извлечения адресов с использованием Flair модели.
        """
        tagger = self.get_ner_model()
        sentence = Sentence(text)
        tagger.predict(sentence)
        result = []
        for span in sentence.get_spans("ner"):
            result.append({"text": span.text, "label": span.tag})
        return result

    async def extract_addresses(self, text: str):
        """
        Асинхронная обертка для функции извлечения адресов.
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, self.extract_addresses_sync, text)
        return result

    @staticmethod
    def process_single_text(text: str, osm_id: int, device: str = "cpu") -> gpd.GeoDataFrame:
        """
        Обрабатывает один текст: извлекает упоминания адресов с помощью модели NER,
        производит предобработку и геокодирование через класс Geocoder.
        
        Аргументы:
            text (str): Текст для обработки.
            osm_id (int): OpenStreetMap ID города/региона, используемый для получения OSM-данных.
            device (str): Устройство для вычислений ("cpu" или "cuda"), по умолчанию "cpu".
            
        Возвращает:
            gpd.GeoDataFrame: Геоданные с результатами геокодирования.
        """
        # Создаём DataFrame с одним текстом
        df = pd.DataFrame({"text": [text]})
        
        # Инициализируем Geocoder с переданным DataFrame и osm_id
        geocoder = Geocoder(
            df=df,
            osm_id=osm_id,
            device=device,  # устройство для модели Flair
            model_path="Geor111y/flair-ner-addresses-extractor",  # путь к модели NER
            text_column_name="text"
        )
        
        # Запускаем обработку (извлечение адресов, геокодирование, объединение данных)
        result_gdf = geocoder.run()
        result_gdf = result_gdf[['Location', 'geometry']]
        return json.loads(result_gdf.to_json())

    @staticmethod
    def fuzzy_search(text, phrase, threshold=80):
        """
        Ищет фразу в тексте с использованием нестрогого сравнения.
        Возвращает True, если найден фрагмент текста, схожий с phrase по уровню similarity >= threshold.
        """
        text = text.lower()
        phrase = phrase.lower()
        words = text.split()
        phrase_words = phrase.split()
        n = len(phrase_words)
        if n == 0:
            return False
        for i in range(len(words) - n + 1):
            window = " ".join(words[i:i+n])
            if fuzz.ratio(phrase, window) >= threshold:
                return True
        return False

    @staticmethod
    def replace_service_names(services_list, ru_service_names):
        inverted = {eng: rus for rus, eng in ru_service_names.items()}
        return [inverted.get(service, service) for service in services_list]

    async def detect_services(self, text):
        """
        Возвращает список сервисов (названий ключей), упомянутых в тексте.
        Учитываются:
        – базовые ключевые слова из keywords_dict,
        – исключаются совпадения, попадающие в irrelevant_mentions_dict,
        – применяются правила из priority_and_exact_keywords_dict.
        """
        detected_services = []
        text_lower = text.lower()
        service_keywords = CONSTANTS.json['service_keywords']
        service_irrelevant_mentions = CONSTANTS.json['service_irrelevant_mentions']
        services_priority_and_exact_keywords = CONSTANTS.json['services_priority_and_exact_keywords']
        ru_service_names = CONSTANTS.json['ru_service_names']
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
                exact_found = any(self.fuzzy_search(text_lower, ex_kw, threshold=85) for ex_kw in local_config.get('exact_keywords', []))
                if exact_found:
                    detected_services.append(service)
                    continue

                priority_over_found = any(self.fuzzy_search(text_lower, po_kw, threshold=85) for po_kw in local_config.get('priority_over', []))
                if priority_over_found:
                    continue
                for excl in local_config.get('exclude_verbs', []):
                    if self.fuzzy_search(text_lower, excl, threshold=85):
                        found = False
                        break
                if not found:
                    continue

            detected_services.append(service)
        detected_services = self.replace_service_names(detected_services, ru_service_names)
        return detected_services

    async def detect_indicators(self, text):
        """
        Возвращает список показателей (названий ключей), упомянутых в тексте.
        Учитываются:
        – базовые ключевые слова из keywords_dict,
        – исключаются совпадения, попадающие в irrelevant_mentions_dict,
        – применяются правила из priority_and_exact_keywords_dict.
        """
        detected_indicators = []
        text_lower = text.lower()
        indicators_keywords = CONSTANTS.json['indicators_keywords']
        indicators_irrelevant_mentions = CONSTANTS.json['indicators_irrelevant_mentions']
        indicators_priority_and_exact_keywords = CONSTANTS.json['indicators_priority_and_exact_keywords']
        for indicator, keywords in indicators_keywords.items():
            found = False
            for kw in keywords:
                if self.fuzzy_search(text_lower, kw):
                    found = True
                    break
            if not found:
                continue

            for irr in indicators_irrelevant_mentions.get(indicator, []):
                if self.fuzzy_search(text_lower, irr):
                    found = False
                    break
            if not found:
                continue

            if indicator in indicators_priority_and_exact_keywords:
                local_config = indicators_priority_and_exact_keywords[indicator]
                exact_found = any(self.fuzzy_search(text_lower, ex_kw, threshold=85) for ex_kw in local_config.get('exact_keywords', []))
                if exact_found:
                    detected_indicators.append(indicator)
                    continue

                priority_over_found = any(self.fuzzy_search(text_lower, po_kw, threshold=85) for po_kw in local_config.get('priority_over', []))
                if priority_over_found:
                    continue
                for excl in local_config.get('exclude_verbs', []):
                    if self.fuzzy_search(text_lower, excl, threshold=75):
                        found = False
                        break
                if not found:
                    continue

            detected_indicators.append(indicator)
        return detected_indicators

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
        # Приводим context к строке, если это список
        context_str = "\n".join(context) if isinstance(context, list) else str(context)
        dict_example = {
            "name": "Юбилейный",
            "notes": "Исторический ресторан на окраине города",
            "location": "Дубровка, Россия"
        }
        prompt = f'''
            Найди названия в тексте {context_str}. 
            Названия должны принадлежать объектам или организациям, которые физически представлены на территории.
            Приведи названия в начальную форму. Добавь как можно больше названий на основе контекста.
            Если в тексте нет упоминаний местоположения, это Ленинграская область.
            Преобразуй название, краткое описание на основе контекста и местоположение в словарь для поиска в OSM.
            Словарь должен быть корректной формы.
            Пример итогового словаря: {dict_example}
            Если названий больше одного, сохрани словари элементами в списке через запятую [dict1, dict2, dict3]
            '''
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
            "temperature": 0.2,
            "prompt": prompt,
            "stream": False,
        }

        def sync_request():
            logger.info("Отправка запроса на URL: %s", self.url)
            try:
                response = requests.post(
                    self.url,
                    headers=headers,
                    json=data,
                    cert=(self.client_cert, self.client_key),
                    verify=self.ca_cert
                )
                if response.status_code == 200:
                    logger.info("Получен успешный ответ от модели (код %s).", response.status_code)
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
        logger.info("Начало обработки описаний для %d элементов.", len(items))
        tasks = [self.describe_async(item['context']) for item in items]
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
        logger.info("Начало разделения столбца 'extracted_data' на 'logic' и 'response'.")

        def extract_think(text):
            match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
            return match.group(1).strip() if match else ""

        def extract_response(text):
            text_without_think = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            code_match = re.search(r'```(?:\w+)?\s*(.*?)\s*```', text_without_think, re.DOTALL)
            if code_match:
                return code_match.group(1).strip()
            else:
                return text_without_think

        df['logic'] = df['extracted_data'].apply(extract_think)
        df['response'] = df['extracted_data'].apply(extract_response)
        logger.info("Разделение завершено.")
        return df

    def parse_response(self, response_str):
        """
        Извлекает словарь или список словарей из строки 'response'.
        """
        logger.debug("Начало парсинга ответа.", response_str)
        cleaned_str = response_str.strip()

        if not cleaned_str:
            logger.warning("Пустая строка ответа после очистки.")
            return {}

        if cleaned_str.startswith('(') and cleaned_str.endswith(')'):
            cleaned_str = '[' + cleaned_str[1:-1] + ']'
            logger.debug("Обнаружены кортежные скобки. Заменены на квадратные", cleaned_str)

        if cleaned_str.startswith('[') and cleaned_str.endswith(']'):
            try:
                result = ast.literal_eval(cleaned_str)
                if isinstance(result, list):
                    logger.debug("Извлечен список словарей через ast.literal_eval.")
                    return result
                else:
                    logger.warning("Ожидался список словарей, но получен другой тип данных.")
                    return {}
            except Exception as e:
                logger.error("Ошибка при разборе списка словарей", e)
                return {}

        match = re.search(r'(\{.*\})', cleaned_str, re.DOTALL)
        if match:
            dict_str = match.group(1)
            try:
                result = ast.literal_eval(dict_str)
                logger.debug("Парсинг словаря через ast.literal_eval успешен.")
                return result
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
            if "429" in error_message or "too many requests" in error_message:
                return "limit_exceeded"
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

    def combine_tags(self, row):
        """
        Объединяет теги OSM в список строк вида "класс:тип".
        """
        if row['osm_class'] is None or row['osm_type'] is None:
            return None
        return [f"{cls}:{typ}" for cls, typ in zip(row['osm_class'], row['osm_type'])]

    def unique_list(self, x):
        return list(dict.fromkeys(x))

    async def process_texts(self, texts: pd.DataFrame) -> gpd.GeoDataFrame:
        """
        Основной процесс обработки DataFrame texts:
         - отправка запросов,
         - разделение и парсинг извлечённых данных,
         - геокодирование и агрегирование.
        """
        items = [{'context': row['text']} for _, row in texts.iterrows()]
        descriptions = await self.process_write_descriptions(items)
        texts['extracted_data'] = descriptions
        texts = self.split_extracted_data(texts)
        texts['response'] = texts['response'].apply(self.parse_response)
        texts['response'] = texts['response'].map(self.fix_response)
        texts = texts[texts['response'].map(lambda x: len(x)) > 0]

        named_objects = texts.explode('response')
        named_objects = named_objects[['text', 'Location', 'geometry', 'logic', 'response']]
        named_objects['object_name'] = named_objects['response'].map(lambda x: x.get("name", None))
        named_objects['object_location'] = named_objects['response'].map(lambda x: x.get("location", None))
        named_objects['object_description'] = named_objects['response'].map(lambda x: x.get("notes", None))
        named_objects['query'] = named_objects['object_name'] + ', ' + named_objects['object_location']
        
        named_objects['query_result'] = named_objects['query'].progress_map(lambda x: self.safe_geocode_with_tags(x))
        logger.info('Data from OSM collected')
        named_objects.rename(columns={'geometry': 'street_geometry'}, inplace=True)
        parsed_df = pd.json_normalize(named_objects.query_result)
        parsed_df = parsed_df.drop(columns=['bbox_north', 'bbox_south', 'bbox_east', 'bbox_west', 'lat', 'lon'], errors='ignore')
        parsed_df = gpd.GeoDataFrame(parsed_df, geometry='geometry').set_crs(4326)
        parsed_df['geometry'] = parsed_df['geometry'].to_crs(3857).centroid.to_crs(4326)

        named_objects = pd.concat([named_objects.reset_index(drop=True), parsed_df], axis=1)
        named_objects['geometry'] = named_objects['geometry'].fillna(gpd.GeoSeries.from_wkt(named_objects['street_geometry']))
        named_objects.drop(columns=['response', 'object_location', 'index', 'logic', 'query', 'place_id', 'query_result', 
                                    'street_geometry', 'osm_type', 'importance', 'place_rank', 'addresstype', 'name'], 
                             inplace=True, errors='ignore')
        named_objects.rename(columns={'Location': 'street_location', 'class': 'osm_class', 'type': 'osm_type', 
                                      'display_name': 'osm_name', 'count': 'message_count'}, inplace=True)
        named_objects = gpd.GeoDataFrame(named_objects, geometry='geometry').set_crs(4326)
        named_objects = named_objects[~named_objects['osm_type'].isin(['administrative', 'city', 'government', 'town', 
                                                                    'townhall', 'courthouse', 'quarter'])]

        logger.info(f'Named objects processed')

        grouped_df = named_objects.groupby(['geometry', 'object_name'], as_index=False).agg(self.unique_list)
        group_counts = named_objects.groupby(['geometry', 'object_name']).size().reset_index(name='count')
        grouped_df = pd.merge(grouped_df, group_counts, on=['geometry', 'object_name'])
        grouped_df = gpd.GeoDataFrame(grouped_df, geometry='geometry').set_crs(4326)
        logger.info(f'Named objects grouped')

        grouped_df['osm_id'] = self.replace_nan_in_column(grouped_df['osm_id'])
        grouped_df['osm_class'] = self.replace_nan_in_column(grouped_df['osm_class'])
        grouped_df['osm_type'] = self.replace_nan_in_column(grouped_df['osm_type'])
        grouped_df['osm_name'] = self.replace_nan_in_column(grouped_df['osm_name'])

        grouped_df['osm_tag'] = grouped_df.apply(self.combine_tags, axis=1)
        grouped_df.drop(columns=['osm_class', 'osm_type'], inplace=True, errors='ignore')
        grouped_df = grouped_df[~grouped_df.object_name.isin(['Александр Дрозденко', 'Игорь Самохин'])]
        return json.loads(grouped_df.to_json())


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
        prompt = f'''
            Найди обсуждение показателей в тексте {context_str}. 
            Строительство - обращение упоминает строительство новых объектов, реновацию или реконструкцию, открытие объектов, постройку.
            Снос - обращение упоминает уничтожение объектов, их разрушение, повреждение. Любая утрата объекта или его части умышленным образом. 
            Обеспеченность - обращение упоминает то, насколько сервис(объект) загружен жителями. Признаки чрезмерной загруженности - очереди, нехватка мест, нехватка персонала.
            Доступность - обращение упоминает сложности с достижением объекта или сервиса в разумное время. Слишком долго или сложно добираться, слишком большое расстояние до ближайшего объекта. 
            В ответе должен быть список из упомянутых показателей в формате [ind1, ind2, ind3]. Если показатели не обсуждаются, пиши []. Не пиши ничего, кроме списка. Нужен ответ правильного формата.
            '''
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
                    verify=self.ca_cert
                )
                if response.status_code == 200:
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

    async def process_find_indicators(self, items):
        """
        Обрабатывает список словарей с ключом 'context' и возвращает список результатов.
        """
        tasks = [self.describe_async(item['context']) for item in items]
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
        if s.startswith('[') and s.endswith(']'):
            inner = s[1:-1].strip()
        else:
            inner = s
        if not inner:
            return []
        return [word.strip().capitalize() for word in inner.split(',') if word.strip()]

    async def get_indicators(self, df):
        """
        Принимает DataFrame с колонкой 'text', обрабатывает запросы и возвращает Series с результатами.
        """
        items = [{'context': row['text']} for _, row in df.iterrows()]
        indicators = await self.process_find_indicators(items)
        df['indicators'] = indicators
        df['indicators'] = df['indicators'].map(self.parse_indicator_response)
        return df['indicators'].tolist()


preprocessing = Preprocessing()
indicators_definition = IndicatorDefinition()
ner_extraction = NER_EXTRACTOR()