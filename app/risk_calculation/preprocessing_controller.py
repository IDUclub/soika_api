from fastapi import APIRouter, HTTPException
from loguru import logger
from app.risk_calculation.logic.preprocessing_methods import preprocessing, ner_extraction, indicators_definition
from app.risk_calculation.dto.vk_requests_dto import VKGroupsRequest, VKTextsRequest
import pandas as pd #костыль
preprocessing_router = APIRouter()

@preprocessing_router.get("/determine_emotion")
async def predict(text: str):
    """
    Эндпоинт для классификации эмоций.
    Принимает параметр text и возвращает результат работы модели классификации.
    """
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Параметр text не может быть пустым")
    result = await preprocessing.classify_emotion(text)
    return {"emotion": result}

@preprocessing_router.get("/extract_addresses")
async def extract_addresses(text: str, osm_id:int):
    """
    Эндпоинт для извлечения адресов из текста.
    Принимает параметр text и возвращает результат работы NER модели.
    """
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Параметр text не может быть пустым")
    result = preprocessing.process_single_text(text, osm_id)
    logger.info("Extraction of addresses completed")
    return {"result": result}

@preprocessing_router.post("/collect_vk_groups")
async def collect_vk_groups(data: VKGroupsRequest):
    result = preprocessing.search_vk_groups(data.territory_name)
    logger.info(f"VK groups for {data.territory_name} collected")
    return {"groups": result}

@preprocessing_router.post("/collect_vk_texts")
async def collect_vk_texts(data: VKTextsRequest):
    result = preprocessing.parse_VK_texts(data.group_domains, data.to_date)
    logger.info(f"VK texts for {data.group_domains} collected")
    return {"texts": result}

@preprocessing_router.get("/extract_named_objects")
async def extract_named_objects(text: str, location: str = "Ленинградская область"):
    """
    GET-метод для обработки одной строки.
    Параметры:
      - text: текст для обработки.
      - location: местоположение (по умолчанию "Ленинградская область").
    """
    df = pd.DataFrame({
        "text": [text],
        "Location": [location],
        "geometry": [None]
    })
    result_gdf = await ner_extraction.process_texts(df)
    return {"result": result_gdf}

@preprocessing_router.get("/extract_services")
async def extract_services(text: str):
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Параметр text не может быть пустым")
    result = await preprocessing.detect_services(text)
    return {"services":result}

@preprocessing_router.get("/extract_indicators")
async def define_indicators(text: str):
    df = pd.DataFrame({
        "text": [text],
    })
    result = await indicators_definition.get_indicators(df)
    return {"indicators":result}