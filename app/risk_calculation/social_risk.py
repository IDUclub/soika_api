from loguru import logger
from app.risk_calculation.modules.social_risk import risk_calculation
from app.risk_calculation.modules.coverage import coverage_calculation
from app.risk_calculation.modules.risk_values import risk_values_collection
from app.risk_calculation.modules.risk_provision import risk_provision_collection
from app.risk_calculation.modules.texts_processing import text_processing
from app.risk_calculation.modules.named_objects import named_objects_collection
from app.risk_calculation.modules.effects import effects_calculation

class RiskCalculationService:
    @staticmethod
    async def get_social_risk(territory_id: int, project_id: int, token: str) -> dict:
        logger.info(f"Service: Calculating social risk for territory {territory_id} and project {project_id}")
        return await risk_calculation.calculate_social_risk(territory_id, project_id, token)

    @staticmethod
    async def get_risk_coverage(territory_id: int, project_id: int, token: str) -> dict:
        logger.info(f"Service: Calculating risk coverage for territory {territory_id} and project {project_id}")
        return await coverage_calculation.calculate_coverage(territory_id, project_id, token)

    @staticmethod
    async def collect_texts(territory_id: int, project_id: int, time_period: str, token: str) -> dict:
        logger.info(f"Service: Collecting texts for territory {territory_id}, project {project_id} for period {time_period}")
        return await text_processing.collect_texts(territory_id, project_id, time_period, token)

    @staticmethod
    async def generate_risk_values(territory_id: int, project_id: int, token: str) -> dict:
        logger.info(f"Service: Generating risk values for territory {territory_id} and project {project_id}")
        return await risk_values_collection.calculate_values_to_risk_data(territory_id, project_id, token)

    @staticmethod
    async def generate_risk_provision(territory_id: int, project_id: int, token: str) -> dict:
        logger.info(f"Service: Generating risk provision for territory {territory_id} and project {project_id}")
        return await risk_provision_collection.calculate_provision_to_risk_data(territory_id, project_id, token)

    @staticmethod
    async def get_named_objects(territory_id: int, project_id: int, token: str) -> dict:
        logger.info(f"Service: Collecting named objects for territory {territory_id} and project {project_id}")
        return await named_objects_collection.collect_named_objects(territory_id, project_id, token)

    @staticmethod
    async def get_risk_effects(territory_id: int, project_id: int, scenario_id: int, token: str) -> dict:
        logger.info(f"Service: Calculating risk effects for territory {territory_id}, project {project_id}, scenario {scenario_id}")
        return await effects_calculation.calculate_risk_for_effects(territory_id, project_id, scenario_id, token)
