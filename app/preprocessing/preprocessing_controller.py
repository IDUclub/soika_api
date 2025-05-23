from fastapi import APIRouter, UploadFile, Request, File, Query, status, BackgroundTasks
from fastapi.responses import JSONResponse
from app.preprocessing.preprocessing import PreprocessingService
from app.preprocessing.dto.vk_requests_dto import VKGroupsRequest, VKTextsRequest
from app.preprocessing.dto.territory_dto import TerritoryCreate
from app.preprocessing.dto.emotion_dto import EmotionCreate
from app.preprocessing.dto.indicator_dto import IndicatorCreate

from app.schema.common_response import DetailResponse, StatusResponse
from app.schema.territories_response import TerritoriesListResponse, CreateTerritoryResponse
from app.schema.groups_response import GroupsListResponse
from app.schema.messages_response import (
    MessagesListResponse,
    UploadMessagesResponse,
    VKTextsResponse,
    CreateEmotionResponse,
    AddressesExtractionResponse
)
from app.schema.named_objects_response import (
    NamedObjectsListResponse,
    UploadNamedObjectsResponse,
    ExtractNamedObjectsResponse
)
from app.schema.indicators_response import (
    IndicatorsListResponse,
    MessageIndicatorPairsResponse,
    CreateIndicatorResponse,
    ExtractIndicatorsResponse
)
from app.schema.services_response import (
    ServicesListResponse,
    MessageServicePairsResponse,
    ExtractServicesResponse
)

territories_router = APIRouter()
groups_router = APIRouter()
messages_router = APIRouter()
named_objects_router = APIRouter()
indicators_router = APIRouter()
services_router = APIRouter()

# Territories endpoints
@territories_router.get("/get_territories", response_model=TerritoriesListResponse)
async def get_territories() -> TerritoriesListResponse:
    return await PreprocessingService.get_territories()

@territories_router.post(
    "/add_territory",
    status_code=201,
    response_model=CreateTerritoryResponse
)
async def create_territory(payload: TerritoryCreate) -> CreateTerritoryResponse:
    return await PreprocessingService.create_territory(payload)

@territories_router.delete("/territories", response_model=DetailResponse)
async def delete_territories() -> DetailResponse:
    return await PreprocessingService.delete_territories()

# Groups endpoints
@groups_router.get("/get_groups", response_model=GroupsListResponse)
async def get_groups() -> GroupsListResponse:
    return await PreprocessingService.get_groups()

@groups_router.post(
    "/collect_vk_groups",
    status_code=201,
    response_model=StatusResponse
)
async def collect_vk_groups(data: VKGroupsRequest) -> StatusResponse:
    return await PreprocessingService.collect_vk_groups(data)

@groups_router.delete("/groups", response_model=DetailResponse)
async def delete_groups() -> DetailResponse:
    return await PreprocessingService.delete_groups()

# Messages endpoints
@messages_router.get("/get_messages", response_model=MessagesListResponse)
async def get_messages(
    only_with_location: bool = Query(
        False, description="Возвращать только записи с имеющимися geometry и location"
    )
) -> MessagesListResponse:
    return await PreprocessingService.get_messages(only_with_location)

@messages_router.post(
    "/add_messages",
    status_code=201,
    response_model=UploadMessagesResponse
)
async def add_messages(file: UploadFile = File(...)) -> UploadMessagesResponse:
    return await PreprocessingService.add_messages(file)

@messages_router.post(
    "/collect_vk_texts",
    status_code=201,
    response_model=VKTextsResponse
)
async def collect_vk_texts(data: VKTextsRequest, request: Request) -> VKTextsResponse:
    return await PreprocessingService.collect_vk_texts(data, request=request)

@messages_router.post(
    "/add_emotions",
    status_code=201,
    response_model=CreateEmotionResponse
)
async def add_emotions(payload: EmotionCreate) -> CreateEmotionResponse:
    return await PreprocessingService.add_emotions(payload)

@messages_router.post(
    "/determine_emotion",
    status_code=201,
    response_model=DetailResponse
)
async def determine_emotion() -> DetailResponse:
    return await PreprocessingService.determine_emotion()

@messages_router.post(
    "/extract_addresses",
    status_code=201,
    response_model=AddressesExtractionResponse
)
async def extract_addresses(
    device: str = "cpu",
    top: int = Query(None, description="Сколько сообщений обрабатывать (None = все)"),
    territory_name: str = Query("Ленинградская область", description="Название региона для геокодирования")
) -> AddressesExtractionResponse:
    return await PreprocessingService.extract_addresses(device, top, territory_name)

@messages_router.delete("/messages", response_model=DetailResponse)
async def delete_messages() -> DetailResponse:
    return await PreprocessingService.delete_messages()

# Named Objects endpoints
@named_objects_router.get("/named_objects", response_model=NamedObjectsListResponse)
async def named_objects() -> NamedObjectsListResponse:
    return await PreprocessingService.get_named_objects()

@named_objects_router.post(
    "/add_named_objects",
    status_code=201,
    response_model=UploadNamedObjectsResponse
)
async def add_named_objects(file: UploadFile = File(...)) -> UploadNamedObjectsResponse:
    return await PreprocessingService.add_named_objects(file)

@named_objects_router.post(
    "/extract_named_objects",
    status_code=201,
    response_model=ExtractNamedObjectsResponse
)
@named_objects_router.post(
    "/extract_named_objects",
    status_code=202
)
async def extract_named_objects(
    background_tasks: BackgroundTasks,
    top: int = Query(None, description="Сколько сообщений обрабатывать за один вызов (None = все)")
) -> JSONResponse:
    """
    Запускает извлечение именованных сущностей в фоне
    и сразу отдаёт клиенту подтверждение.
    """

    background_tasks.add_task(PreprocessingService.extract_named_objects, top)
    return JSONResponse(
        status_code=202,
        content={
            "status": "processing",
            "message": "NER extraction has started in background "
        }
    )

@named_objects_router.delete("/named_objects", response_model=DetailResponse)
async def delete_named_objects() -> DetailResponse:
    return await PreprocessingService.delete_named_objects()

# Indicators endpoints
@indicators_router.get("/get_indicators", response_model=IndicatorsListResponse)
async def get_indicators() -> IndicatorsListResponse:
    return await PreprocessingService.get_indicators()

@indicators_router.get("/get_message_indicator_pairs", response_model=MessageIndicatorPairsResponse)
async def get_message_indicator_pairs() -> MessageIndicatorPairsResponse:
    return await PreprocessingService.get_message_indicator_pairs()

@indicators_router.post(
    "/add_indicators",
    status_code=201,
    response_model=CreateIndicatorResponse
)
async def add_indicators(payload: IndicatorCreate) -> CreateIndicatorResponse:
    return await PreprocessingService.add_indicators(payload)

@indicators_router.post(
    "/extract_indicators",
    status_code=202
)
async def extract_indicators(
    background_tasks: BackgroundTasks,
    top: int = Query(None, description="Сколько сообщений обрабатывать за один вызов (None = все)")
) -> JSONResponse:
    """
    Запускает извлечение индикаторов в фоне и сразу возвращает статус обработки.
    """
    background_tasks.add_task(PreprocessingService.extract_indicators, top)

    return JSONResponse(
        status_code=202,
        content={
            "status": "processing",
            "message": "Indicator extraction is started in the background"
        }
    )

@indicators_router.delete("/indicators", response_model=DetailResponse)
async def delete_indicators() -> DetailResponse:
    return await PreprocessingService.delete_indicators()

# Services endpoints
@services_router.get("/get_services", response_model=ServicesListResponse)
async def get_services() -> ServicesListResponse:
    return await PreprocessingService.get_services()

@services_router.get("/get_message_service_pairs", response_model=MessageServicePairsResponse)
async def get_message_service_pairs() -> MessageServicePairsResponse:
    return await PreprocessingService.get_message_service_pairs()

@services_router.post(
    "/extract_services",
    status_code=201,
    response_model=ExtractServicesResponse
)
async def extract_services(
    top: int = Query(None, description="Сколько сообщений обрабатывать за один вызов (None = все)")
) -> ExtractServicesResponse:
    return await PreprocessingService.extract_services(top)
