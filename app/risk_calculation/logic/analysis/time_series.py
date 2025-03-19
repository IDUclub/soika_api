import pandas as pd
import numpy as np
from loguru import logger

from app.common.api.urbandb_api_gateway import urban_db_api
from app.risk_calculation.logic.analysis.texts_processing import text_processing
from app.risk_calculation.logic.analysis.constants import CONSTANTS

class TimeSeries:
    async def collect_texts(self, territory_id, project_id, period):
        """
        Retrieves texts for a given project and groups them by a specified period.
        """
        logger.info(f"Retrieving texts for project {project_id} and its context")
        project_area = await urban_db_api.get_context_territories(territory_id, project_id)
        texts = await text_processing.get_texts(project_area)
        if len(texts) == 0:
            logger.info("No texts for this area")
            return {}

        services_categories = CONSTANTS.json['services_categories']
        texts['category'] = texts['services'].map(services_categories) + ' инфраструктура'
        missing_services = texts[texts['category'].isna()].services.unique().tolist()
        if missing_services:
            logger.info(f"Attention: services not mapped to categories: {missing_services}")

        texts = texts.replace({np.nan: None})
        texts["date"] = pd.to_datetime(texts["date"])

        if period == 'day':
            texts['period'] = texts['date'].dt.floor('D')
            freq = 'D'
        elif period == 'week':
            texts['period'] = texts['date'].dt.to_period('W').apply(lambda r: r.start_time)
            freq = 'W-MON'
        elif period == 'month':
            texts['period'] = texts['date'].dt.to_period('M').apply(lambda r: r.start_time)
            freq = 'MS'
        elif period == 'year':
            texts['period'] = texts['date'].dt.to_period('Y').apply(lambda r: r.start_time)
            freq = 'AS'
        else:
            raise ValueError("Invalid period specified.")
        
        grouped = texts.groupby(['category', 'period']).size().reset_index(name='count')

        start_date = grouped['period'].min()
        end_date = grouped['period'].max()
        timeline_df = pd.DataFrame({'date': pd.date_range(start=start_date, end=end_date, freq=freq)})
        timeline_df['date'] = timeline_df['date'].dt.strftime('%Y-%m-%d')
        timeline_data = timeline_df.to_dict(orient='records')

        grouped.rename(columns={'period': 'date'}, inplace=True)
        grouped['date'] = grouped['date'].dt.strftime('%Y-%m-%d')
        data = grouped.to_dict(orient='records')

        return {'timeline': timeline_data, 'data': data}

time_series = TimeSeries()