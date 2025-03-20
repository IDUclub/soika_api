import pandas as pd
import numpy as np

from loguru import logger
from app.risk_calculation.logic.analysis.social_risk import risk_calculation
from app.risk_calculation.logic.analysis.texts_processing import text_processing
from app.risk_calculation.logic.analysis.constants import CONSTANTS
from app.common.api.urbandb_api_gateway import urban_db_api
from app.common.api.townsnet_api_gateway import townsnet_api

class RiskProvision:
    async def calculate_provision_to_risk_data(self, territory_id, project_id, token):
        logger.info(f"Retrieving texts for project {project_id} and its context")
        project_area = await urban_db_api.get_context_territories(territory_id, project_id)
        texts = await text_processing.get_texts(project_area)
        
        if len(texts) == 0:
            logger.info(f"No texts for this area")
            response = {}
            return response
        logger.info(f"Calculating risk-to-value table for project {project_id} and its context")
        scored_texts = await risk_calculation.calculate_score(texts)
        score_df = await text_processing.summarize_risk(scored_texts)

        territories_with_provision = await townsnet_api.get_evaluated_territories(project_id, token)
        services = [col for col in territories_with_provision.columns if col not in ['geometry', 'territory_id', 'basic', 'additional', 'comfort', 'Обеспеченность']]
        provision = {
            'service': services,
            'median': [territories_with_provision[service].median() for service in services]
        }
        provision = pd.DataFrame(provision).set_index('service')
        combined_df = provision.join(score_df)[['total_score', 'median']]
        combined_df.columns = ['risk', 'provision']
        combined_df['risk'] = combined_df['risk'].fillna(0).round(2)
        combined_df['provision'] = combined_df['provision'].round(2)
        combined_df['category'] = combined_df.index.map(CONSTANTS.json['services_categories'])
        combined_df['category'] = combined_df['category'] + ' инфраструктура'
        combined_df.reset_index(inplace=True)
        combined_df = combined_df.replace({np.nan: None})        
        response = {'provision_to_risk_table': combined_df.to_dict(orient='records')}
        logger.info(f"Table response generated")
        return response
    
risk_provision_collection = RiskProvision()