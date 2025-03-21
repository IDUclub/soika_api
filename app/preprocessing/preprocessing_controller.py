from fastapi import APIRouter, UploadFile, File, Query
from app.preprocessing.preprocessing import PreprocessingService

territories_router = APIRouter()
groups_router = APIRouter()
messages_router = APIRouter()
named_objects_router = APIRouter()
indicators_router = APIRouter()
services_router = APIRouter()

# Territories endpoints

@territories_router.get("/get_territories")
async def get_territories():
    return await PreprocessingService.get_territories()

@territories_router.post("/add_territory")
async def create_territory(payload):
    return await PreprocessingService.create_territory(payload)

@territories_router.delete("/territories")
async def delete_territories():
    return await PreprocessingService.delete_territories()

# Groups endpoints

@groups_router.get("/get_groups")
async def get_groups():
    return await PreprocessingService.get_groups()

@groups_router.post("/collect_vk_groups")
async def collect_vk_groups(data):
    return await PreprocessingService.collect_vk_groups(data)

@groups_router.delete("/groups")
async def delete_groups():
    return await PreprocessingService.delete_groups()

# Messages endpoints

@messages_router.get("/get_messages")
async def get_messages(
    only_with_location: bool = Query(
        False, description="Возвращать только записи с непустыми geometry и location"
    )
):
    return await PreprocessingService.get_messages(only_with_location)

@messages_router.post("/add_messages")
async def add_messages(file: UploadFile = File(...)):
    return await PreprocessingService.add_messages(file)

@messages_router.post("/collect_vk_texts")
async def collect_vk_texts(data):
    return await PreprocessingService.collect_vk_texts(data)

@messages_router.post("/add_emotions")
async def add_emotions(payload):
    return await PreprocessingService.add_emotions(payload)

@messages_router.post("/determine_emotion")
async def determine_emotion():
    return await PreprocessingService.determine_emotion()

@messages_router.post("/extract_addresses")
async def extract_addresses(
    device: str = "cpu",
    top: int = Query(None, description="Сколько сообщений обрабатывать (None = все)"),
    territory_name: str = Query("Ленинградская область", description="Название региона для геокодирования")
):
    return await PreprocessingService.extract_addresses(device, top, territory_name)

@messages_router.delete("/messages")
async def delete_messages():
    return await PreprocessingService.delete_messages()

# Named Objects endpoints

@named_objects_router.get("/named_objects")
async def named_objects():
    return await PreprocessingService.get_named_objects()

@named_objects_router.post("/add_named_objects")
async def add_named_objects(file: UploadFile = File(...)):
    return await PreprocessingService.add_named_objects(file)

@named_objects_router.post("/extract_named_objects")
async def extract_named_objects(
    top: int = Query(None, description="Сколько сообщений обрабатывать за один вызов (None = все)")
):
    return await PreprocessingService.extract_named_objects(top)

@named_objects_router.delete("/named_objects")
async def delete_named_objects():
    return await PreprocessingService.delete_named_objects()

# Indicators endpoints

@indicators_router.get("/get_indicators")
async def get_indicators():
    return await PreprocessingService.get_indicators()

@indicators_router.get("/get_message_indicator_pairs")
async def get_message_indicator_pairs():
    return await PreprocessingService.get_message_indicator_pairs()

@indicators_router.post("/add_indicators")
async def add_indicators(payload):
    return await PreprocessingService.add_indicators(payload)

@indicators_router.post("/extract_indicators")
async def extract_indicators(
    top: int = Query(None, description="Сколько сообщений обрабатывать за один вызов (None = все)")
):
    return await PreprocessingService.extract_indicators(top)

@indicators_router.delete("/indicators")
async def delete_indicators():
    return await PreprocessingService.delete_indicators()

# Services endpoints

@services_router.get("/get_services")
async def get_services():
    return await PreprocessingService.get_services()

@services_router.get("/get_message_service_pairs")
async def get_message_service_pairs():
    return await PreprocessingService.get_message_service_pairs()

@services_router.post("/extract_services")
async def extract_services(
    top: int = Query(None, description="Сколько сообщений обрабатывать за один вызов (None = все)")
):
    return await PreprocessingService.extract_services(top)
