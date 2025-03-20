import pandas as pd
import geopandas as gpd
import asyncio

from loguru import logger
from app.risk_calculation.logic.analysis.social_risk import risk_calculation
from app.risk_calculation.logic.analysis.texts_processing import text_processing
from app.risk_calculation.logic.analysis.constants import CONSTANTS
from app.common.api.urbandb_api_gateway import urban_db_api
from app.common.api.values_api_gateway import values_api

class RiskValues:
    @staticmethod
    async def get_level_4_territory(territory_id):
            territory = await urban_db_api.get_territory(territory_id)
            while territory.iloc[0]['level'] < 4:
                parent_id = territory.iloc[0]['parent_id']
                if parent_id is None:
                    break 
                territory = await urban_db_api.get_territory(parent_id)
            return territory

    @staticmethod
    async def fetch_municipalities(df):      
        territories_to_fetch = df[df['level'] < 4]
        if len(territories_to_fetch) == 0:
            return df
        tasks = [risk_values_collection.get_level_4_territory(territory_id) for territory_id in territories_to_fetch['territory_id']]
        results = await asyncio.gather(*tasks)
        final_df = gpd.GeoDataFrame(pd.concat(results, ignore_index=True), geometry='geometry', crs=4326)
        
        return final_df

    @staticmethod
    def generate_value_names(value):
        value_mapping = {
            "dev": "Ценности развития",
            "soc": "Социальные ценности",
            "bas": "Базовые ценности"
        }

        group_mapping = {
            "comm": "населения",
            "soc_workers": "трудоспособного населения",
            "soc_old": "населения старше трудоспособного возраста",
            "soc_parents": "населения с детьми",
            "loc": "локальной идентичности"
        }

        value_key, group_key = value.split("/")
        return f"{value_mapping.get(value_key, value_key)} {group_mapping.get(group_key, group_key)}"

    @staticmethod
    async def summarize_risk(df):
        score_df = df.groupby(['services', 'indicators'])['score'].mean().unstack(fill_value=0)
        score_df['total_score'] = score_df.sum(axis=1)
        score_df['total_score'] = (score_df['total_score'] / 5).clip(lower=0, upper=1)
        return score_df


    @staticmethod
    async def map_risk_score(df):
        mapping_df = pd.DataFrame(CONSTANTS.json['service_to_values_mapping'])
        result_series = pd.Series()
        for service, row in df.iterrows():
            group = mapping_df[mapping_df['services'] == service]['social_group'].values
            value_type = mapping_df[mapping_df['services'] == service]['values'].values
            
            if len(group) > 0 and len(value_type) > 0:
                social_group = group[0]
                values = value_type[0]
                index_name = f"{values}/{social_group}"
                result_series[index_name] = row['total_score']

        return result_series

    def generate_category_table(self):
        mapping_df = pd.DataFrame(CONSTANTS.json['service_to_values_mapping'])
        mapping_df['category'] = mapping_df['values'] + '/' + mapping_df['social_group']
        mapping_df['category'] =  mapping_df['category'].map(risk_values_collection.generate_value_names)
        category_table = mapping_df.groupby('category').agg({
            'services': lambda x: ', '.join(x.dropna().astype(str))
        })
        return category_table

    async def calculate_values_to_risk_data(self, territory_id, project_id):
        logger.info(f"Retrieving texts for project {project_id} and its context")
        project_area = await urban_db_api.get_context_territories(territory_id, project_id)
        texts = await text_processing.get_texts(project_area)

        if len(texts) == 0:
            logger.info(f"No texts for this area")
            response = {}
            return response

        logger.info(f"Calculating risk-to-value table for project {project_id} and its context")
        scored_texts = await risk_calculation.calculate_score(texts)
        score_df = await risk_values_collection.summarize_risk(scored_texts)
        risk_series = await risk_values_collection.map_risk_score(score_df)

        municipalities = await risk_values_collection.fetch_municipalities(project_area)
        municipal_districts_list = municipalities['parent_id'].unique().tolist()
        value_data = await asyncio.gather(
            *(values_api.get_value_data(tid) for tid in municipal_districts_list)
            )
        series_dict = {
            tid: series.rename(tid)
            for tid, series in zip(municipal_districts_list, value_data)
        }
        value_data = pd.concat(series_dict.values(), axis=1)
        value_series = value_data.mean(axis=1)
        values_to_risk_table = pd.concat([risk_series, value_series], axis=1)
        values_to_risk_table.columns = ['Общественный резонанс', 'Поддержка ценностей']
        values_to_risk_table = values_to_risk_table.round(4)
        values_to_risk_table['Общественный резонанс'] = values_to_risk_table['Общественный резонанс'].fillna(0)
        values_to_risk_table.reset_index(inplace=True)
        values_to_risk_table.rename(columns={"index": "category"}, inplace=True)
        values_to_risk_table['category'] = values_to_risk_table['category'].map(risk_values_collection.generate_value_names)
        values_to_risk_table.dropna(subset='Поддержка ценностей', inplace=True)
        category_table = risk_values_collection.generate_category_table()
        values_to_risk_table = values_to_risk_table.merge(category_table, on='category', how='left')
        values_to_risk_table['services'] = values_to_risk_table['services'].fillna('Нет данных по сервисам')
        response = {'values_to_risk_table': values_to_risk_table.to_dict(orient='records')}
        logger.info(f"Table response generated")
        return response
    
risk_values_collection = RiskValues()