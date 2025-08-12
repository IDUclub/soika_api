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
        effects = effects.merge(score_df[['services', 'total_score']], left_on='name', right_on='services', how='left')
        effects['total_score'].fillna(0, inplace=True)
        services_data = await urban_db_api.get_services_for_groups()
        effects = (
            effects
            .merge(
                services_data,
                left_on='name',  
                right_index=True,
                how='left'
            )
            .drop(columns=['services', 'id'])
        )
        effects.rename(columns={'infrastructure_type':'category'}, inplace=True)
        effects['category'] = effects['category'].map({"basic":"Базовая", "additional":"Дополнительная", "comfort":"Комфортная"})
        effects['category'] = effects['category'] + ' инфраструктура'
        effects.rename(columns={'total_score':'risk'}, inplace=True)
        effects['category'] = effects['category'].replace({np.nan: None})
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