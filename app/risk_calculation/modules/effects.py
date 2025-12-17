from loguru import logger
import numpy as np
from app.common.api.effects_api_gateway import effects_api
from app.common.api.urbandb_api_gateway import urban_db_api
from app.risk_calculation.modules.texts_processing import text_processing
from app.risk_calculation.modules.social_risk import risk_calculation
from app.common.modules.constants import CONSTANTS


class EffectsCalculation:
    async def calculate_risk_for_effects(self, territory_id, project_id, scenario_id, token):
        """
        Calculates risk for services in scenario and merges them with effects.
        """
        logger.info(f"Retrieving texts for project {project_id} and its context")
        project_area = await urban_db_api.get_context_territories(territory_id, project_id, token)
        texts = await text_processing.get_texts(project_area)
        
        if len(texts) == 0:
            logger.info(f"No texts for this area")
            response = {}
            return response
        logger.info(f"Calculating risk-to-value table for project {project_id} and its context")
        scored_texts = await risk_calculation.calculate_score(texts)
        score_df = await text_processing.summarize_risk(scored_texts)
        score_df = score_df.reset_index()
        score_df['total_score'] = score_df['total_score'].round(2)
        effects = await effects_api.get_evaluated_territories_effects(scenario_id, token)
        if effects is None:
            return None
        service_ids = await effects_api.get_services_ids(scenario_id, token)
        service_mapping = await urban_db_api.get_service_mapping()
        service_mapping = service_ids.merge(service_mapping, on='id')
        effects = effects.merge(service_mapping, on='name_en')
        effects = effects.merge(score_df[['services', 'total_score']], left_on='name', right_on='services', how='left')
        effects['total_score'].fillna(0, inplace=True)
        effects.rename(columns={'total_score':'risk'}, inplace=True)
        services_data = await urban_db_api.get_services_for_groups()
        effects =  effects.merge(
                services_data,
                left_on='name',  
                right_index=True,
                how='left'
            )
        effects = effects[['name', 'category', 'before', 'after', 'delta', 'risk', 'social_group']]
        effects['category'] = effects['category'].map({"basic":"Базовая", "additional":"Дополнительная", "comfort":"Комфортная"})
        effects['category'] = effects['category'] + ' инфраструктура'
        effects['category'] = effects['category'].replace({np.nan: None})
        effects.dropna(subset='social_group', inplace=True)
        effects = (
            effects
            .groupby('name', as_index=False)
            .agg(
                before = ('before', 'first'),
                after = ('after', 'first'),
                delta = ('delta', 'first'),
                risk = ('risk', 'first'),
                category  = ('category',  'first'),
                social_group = ('social_group', list),
            )
        )
        response = {'effects':effects.to_dict(orient='records')}

        return response
    
effects_calculation = EffectsCalculation()