from loguru import logger
from fastapi import Request
from app.preprocessing.modules.territories import territories_calculation
from app.preprocessing.modules.groups import groups_calculation
from app.preprocessing.modules.messages import messages_calculation
from app.preprocessing.modules.geocoder import geocoder
from app.preprocessing.modules.services import services_calculation
from app.preprocessing.modules.indicators import indicators_calculation
from app.preprocessing.modules.named_objects import ner_calculation

class PreprocessingService:

    # Territories methods

    @staticmethod
    async def get_territories(scope):
        logger.info("Service: Getting all territories")
        return await territories_calculation.get_all_territories(scope)

    @staticmethod
    async def create_territory(payload):
        logger.info("Service: Creating new territory")
        return await territories_calculation.create_new_territory(payload)

    @staticmethod
    async def delete_territories():
        logger.info("Service: Deleting all territories")
        return await territories_calculation.delete_all_territories_func()

    # Groups methods

    @staticmethod
    async def get_groups():
        logger.info("Service: Getting all groups")
        return await groups_calculation.get_all_groups()

    @staticmethod
    async def collect_vk_groups(data):
        logger.info("Service: Collecting VK groups")
        return await groups_calculation.collect_vk_groups_func(data)

    @staticmethod
    async def delete_groups():
        logger.info("Service: Deleting groups")
        return await groups_calculation.delete_all_groups_func()

    # Messages methods

    @staticmethod
    async def get_messages(only_with_location: bool):
        logger.info("Service: Getting all messages")
        return await messages_calculation.get_all_messages(only_with_location)

    @staticmethod
    async def add_messages(file):
        logger.info("Service: Adding messages from file")
        return await messages_calculation.upload_messages_func(file)

    @staticmethod
    async def collect_vk_texts(data, request: Request):
        logger.info("Service: Collecting VK texts")
        return await messages_calculation.collect_vk_texts_func(data, request=request)

    @staticmethod
    async def add_emotions(payload):
        logger.info("Service: Adding emotions")
        return await messages_calculation.create_emotion_func(payload)

    @staticmethod
    async def determine_emotion(territory_id: int, top: int):
        logger.info("Service: Determining emotion for unprocessed messages")
        return await messages_calculation.determine_emotion_for_unprocessed_messages_func(territory_id=territory_id, top=top)

    @staticmethod
    async def extract_addresses(top: int, territory_name: str, territory_id: int, token: str):
        logger.info("Service: Extracting addresses for unprocessed messages")
        return await geocoder.extract_addresses_from_texts(
           input_territory_name=territory_name, token=token, territory_id=territory_id, top=top   
        )

    @staticmethod
    async def delete_messages():
        logger.info("Service: Deleting all messages")
        return await messages_calculation.delete_all_messages_func()

    # Named Objects methods

    @staticmethod
    async def get_named_objects():
        logger.info("Service: Getting all named objects")
        return await ner_calculation.get_all_named_objects()

    @staticmethod
    async def add_named_objects(file):
        logger.info("Service: Adding named objects from file")
        return await ner_calculation.upload_named_objects_func(file)

    @staticmethod
    async def extract_named_objects(territory_id: int, top: int):
        logger.info("Service: Extracting named objects")
        return await ner_calculation.extract_named_objects_func(territory_id=territory_id, top=top)

    @staticmethod
    async def delete_named_objects():
        logger.info("Service: Deleting all named objects")
        return await ner_calculation.delete_all_named_objects_func()

    # Indicators methods

    @staticmethod
    async def get_indicators():
        logger.info("Service: Getting all indicators")
        return await indicators_calculation.get_all_indicators()

    @staticmethod
    async def get_message_indicator_pairs():
        logger.info("Service: Getting all message-indicator pairs")
        return await indicators_calculation.get_all_message_indicator_pairs()

    @staticmethod
    async def add_indicators(payload):
        logger.info("Service: Adding a new indicator")
        return await indicators_calculation.create_indicator_func(payload)

    @staticmethod
    async def extract_indicators(territory_id:int, top: int):
        logger.info("Service: Extracting indicators")
        return await indicators_calculation.extract_indicators_func(territory_id=territory_id, top=top)

    @staticmethod
    async def delete_indicators():
        logger.info("Service: Deleting all indicators")
        return await indicators_calculation.delete_all_indicators_func()

    # Services methods

    @staticmethod
    async def get_services():
        logger.info("Service: Getting all services")
        return await services_calculation.get_all_services()

    @staticmethod
    async def get_message_service_pairs():
        logger.info("Service: Getting all message-service pairs")
        return await services_calculation.get_all_message_service_pairs()

    @staticmethod
    async def extract_services(territory_id:int, top: int):
        logger.info("Service: Extracting services")
        return await services_calculation.extract_services_func(territory_id=territory_id, top=top)