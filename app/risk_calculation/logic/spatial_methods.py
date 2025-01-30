import pandas as pd
import geopandas as gpd
import json
import asyncio

from loguru import logger
from shapely.geometry import shape, LineString
from geojson_pydantic import MultiPolygon, Polygon, Point
from app.risk_calculation.logic.constants import TEXTS, bucket_name, text_name
from app.common.api.urbandb_api_gateway import urban_db_api

class RiskCalculation:
    def __init__(self, emotion_weights=None):
        """Initialize the class with emotion weights."""
        self.emotion_weights = emotion_weights or {'positive': 1.5, 'neutral': 1, 'negative': 1.5}

    async def expand_rows_by_columns(self, dataframe, columns):
        """
        Expands rows based on specified columns containing comma-separated values.
        
        Args:
            dataframe (pd.DataFrame): Input DataFrame.
            columns (list): List of columns to expand.

        Returns:
            pd.DataFrame: Expanded DataFrame.
        """
        expanded_df = dataframe.copy()
        for column in columns:
            expanded_df[column] = expanded_df[column].str.split(', ')
            expanded_df = expanded_df.explode(column, ignore_index=True)

        return expanded_df

    @staticmethod
    async def to_gdf(geometry: Polygon | Point | MultiPolygon) -> gpd.GeoDataFrame:
        """
        Convert list of coordinates to GeoDataFrame
        """

        if isinstance(geometry, Point):
            gs: gpd.GeoSeries = gpd.GeoSeries(
                [geometry], crs=4326
            ).to_crs(3857)
            buffer = gs.buffer(500)
            gdf = gpd.GeoDataFrame(geometry=buffer, crs=3857)
            gdf.to_crs(4326, inplace=True)
        else:
            geometry = {'geometry': [shape(geometry)]}
            gdf = gpd.GeoDataFrame(geometry, geometry='geometry', crs=4326)

        return gdf

    @staticmethod
    async def get_texts(
        territory_gdf: gpd.GeoDataFrame
    ):
        """
        Retrieves the source texts in the given territory.

        Args:
            territory_gdf (gpd.GeoDataFrame): A GeoDataFrame representing the area of interest.

        Returns:
            Dict[str, Any]: A dictionary with 'buffer_size' and 'texts' keys.
        """
        territory_gdf = territory_gdf.copy()
        texts_gdf = TEXTS.gdf.copy()

        local_crs = territory_gdf.estimate_utm_crs()
        territory_gdf = territory_gdf.to_crs(local_crs)
        texts_gdf = texts_gdf.to_crs(local_crs)

        texts_gdf = texts_gdf[texts_gdf['type'] != 'post']

        local_texts = gpd.clip(texts_gdf, territory_gdf)

        return {
            'texts': local_texts.to_crs(epsg=4326)
        }

    async def calculate_score(self, df):
        df = df.copy()
        df['emotion_weight'] = df['emotion'].map(self.emotion_weights)
        df['minmaxed_views'] = (df['views.count'] - df['views.count'].min()) / (df['views.count'].max() - df['views.count'].min())
        df['minmaxed_likes'] = (df['likes.count'] - df['likes.count'].min()) / (df['likes.count'].max() - df['likes.count'].min())
        df['minmaxed_reposts'] = (df['reposts.count'] - df['reposts.count'].min()) / (df['reposts.count'].max() - df['reposts.count'].min())
        df['engagement_score'] = df.fillna(0).apply(
            lambda row: row['minmaxed_views'] + row['minmaxed_likes'] + row['minmaxed_reposts'] + 1,
            axis=1
        )
        low_threshold = df['engagement_score'].quantile(1/3)
        high_threshold = df['engagement_score'].quantile(2/3)
        df['activity_level'] = df['engagement_score'].apply(
            lambda x: "активным" if x >= high_threshold else ("умеренно активным" if x >= low_threshold else "малоактивным")
        )
        df['score'] = df['engagement_score'] * df['emotion_weight']
        df['score'] = df['score'].round(4)
        return df.drop(columns=['minmaxed_views', 'minmaxed_likes', 'minmaxed_reposts', 'engagement_score'])

    @staticmethod
    def get_top_indicators(row):
        nonzero_values = row[row != 0]
        top_two = nonzero_values.nlargest(2).index.tolist()
        declensions = {
            "Строительство": "строительством",
            "Снос": "сносом",
            "Противоречие": "противоречиями",
            "Доступность": "доступностью",
            "Обеспеченность": "обеспеченностью"
        }
        declined = [declensions.get(ind, ind) for ind in top_two]
        preposition = "со" if declined and declined[0][0] == "с" else "с"
        return f"{preposition} {', '.join(declined)}" if declined else "без явных индикаторов"

    @staticmethod
    def generate_description(row):
        service_name = row.name
        risk_level = row["risk_level"]
        top_indicators = row["top_indicators"]
        activity_level = row["activity_level"]
        emotion = row["emotion"]

        risk_text = f"Сервис «{service_name}» имеет {risk_level} степень общественного резонанса."
        indicators_list = top_indicators.split(', ')
        indicator_count = len(indicators_list)
        if indicator_count == 1:
            indicators_text = (
                f"Среди показателей выделяется один ключевой — {indicators_list[0]}."
            )
        elif 2 <= indicator_count <= 4:
            indicators_text = (
                f"Среди показателей можно отметить {indicator_count} основных: "
                f"{', '.join(indicators_list)}."
            )
        else:
            indicators_text = (
                f"Среди показателей выделяется несколько основных (всего {indicator_count}), "
                f"включая {', '.join(indicators_list[:2])}."
            )
        activity_text = (
            f"Уровень распространения информации является {activity_level}."
        )
        emotion_mapping = {
            "negative": "негативную",
            "positive": "положительную",
            "neutral": "нейтральную"
        }
        emotion_descr = emotion_mapping.get(emotion, emotion)
        emotion_text = (
            f"Эмоциональную окраску дискуссии можно охарактеризовать как преимущественно {emotion_descr}."
        )
        base_priorities = {
            "indicators": 2,  
            "activity": 1,
            "emotion": 1
        }

        if indicator_count > 1:
            base_priorities["indicators"] += 2
        if emotion in ["negative", "positive"]:
            base_priorities["emotion"] += 2  
        if activity_level == "активным":
            base_priorities["activity"] += 2  
        if indicator_count == 1 and emotion == "negative":
            if base_priorities["emotion"] <= base_priorities["indicators"]:
                base_priorities["emotion"] = base_priorities["indicators"] + 1

        other_blocks = {
            "indicators": (base_priorities["indicators"], indicators_text),
            "activity": (base_priorities["activity"], activity_text),
            "emotion": (base_priorities["emotion"], emotion_text)
        }
        sorted_other_blocks = sorted(
            other_blocks.items(),
            key=lambda x: x[1][0],
            reverse=True
        )
        final_description = " ".join(
            [risk_text] + [block[1][1] for block in sorted_other_blocks]
        )
        return final_description
    
    @staticmethod
    async def score_table(df):
        score_df = df.groupby(['services', 'indicators'])['score'].mean().unstack(fill_value=0)
        score_df_numeric = score_df.copy()
        score_df["top_indicators"] = score_df_numeric.apply(risk_calculator.get_top_indicators, axis=1)
        score_df['risk_rating'] = score_df_numeric.sum(axis=1).clip(upper=5).round(0).astype(int)
        score_df['risk_level'] = score_df['risk_rating'].map(lambda x: "высокую" if x >= 4 else ("среднюю" if 2 <= x < 4 else "низкую"))
    
        emotion_table = df.groupby('services')['emotion'].agg(lambda x: x.mode().iloc[0])
        activity_translation = {
            "активным": "активным",
            "умеренно активным": "умеренным",
            "малоактивным": "слабым"
        }
        activity_table = df.groupby('services')['activity_level'].agg(lambda x: x.mode().iloc[0])
        
        final_table = score_df.join(emotion_table)
        final_table = final_table.join(activity_table)
        final_table["activity_level"] = final_table["activity_level"].map(activity_translation)
        final_table["description"] = final_table.apply(risk_calculator.generate_description, axis=1)
        final_table = final_table[['risk_rating', 'description']]
        return final_table

    async def calculate_social_risk(self, territory_id, project_id):
        logger.info(f"Retrieving texts for project {project_id} and its context")
        project_area = await urban_db_api.get_context_territories(territory_id, project_id)
        texts = await risk_calculator.get_texts(project_area)

        if len(texts['texts']) == 0:
            logger.info(f"No texts for this area")
            response = {}
            return response

        logger.info(f"Calculating social risk for project {project_id} and its context")
        scored_texts = await risk_calculator.calculate_score(texts['texts'])
        score_df = await risk_calculator.score_table(scored_texts)

        texts_df = texts['texts'].copy()
        texts_df = texts_df[['text', 'services', 'indicators']]
        texts_df = texts_df.groupby(
            ['text', 'services'] 
        ).agg({ 
            'indicators': lambda x: ', '.join(set(x))
        }).reset_index()
        result_df = texts_df.groupby('services').agg({
            'text': lambda x: list(x),
            'indicators': lambda x: list(x)
        }).reset_index()
        merged_df = score_df.merge(result_df, on='services', how='left')

        result_dict = merged_df.to_dict(orient='records')

        response = {'social_risk_table': result_dict}
        logger.info(f"Table response generated")
        return response

    @staticmethod
    async def get_areas(urban_areas: gpd.GeoDataFrame, texts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        urban_areas = urban_areas.merge(texts['best_match'].value_counts().rename('count'), left_on='name', right_index=True, how='left')
        urban_areas = urban_areas[['name', 'territory_id', 'admin_center', 'is_city', 'geometry', 'count']]
        urban_areas.dropna(subset='count', inplace=True)
        local_crs = urban_areas.estimate_utm_crs()
        urban_areas['area'] = urban_areas.to_crs(local_crs).area
        urban_areas = urban_areas.sort_values(by='area', ascending=False).drop_duplicates(subset='name', keep='first')
        urban_areas.drop(columns=['area'], inplace=True)
        return urban_areas

    @staticmethod
    async def get_area_centroid(area, region_territories):
        if area['is_city']:
            return area.geometry.centroid
        else:
            filtered_region = region_territories.loc[region_territories['territory_id'] == area.admin_center]
            return filtered_region.geometry.centroid.iloc[0]

    @staticmethod
    async def get_links(project_id: int, urban_areas: gpd.GeoDataFrame, region_territories: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        project_centroid = await urban_db_api.get_project_territory_centroid(project_id)
        areas = urban_areas.copy()
        areas['geometry'] = await asyncio.gather(*[
            risk_calculator.get_area_centroid(area, region_territories) for _, area in areas.iterrows()
        ])
        areas['geometry'] = areas.apply(
            lambda area: LineString([area.geometry, project_centroid]), axis=1
        )
        grouped = areas.groupby('geometry').agg({
            'name': lambda x: list(x)
        }).reset_index()
        grouped.rename(columns={'name': 'urban_area'}, inplace=True)
        combined_lines_gdf = gpd.GeoDataFrame(grouped, geometry='geometry', crs=areas.crs)
        return combined_lines_gdf

    async def calculate_coverage(self, territory_id, project_id):
        logger.info(f"Retrieving texts for project {project_id} and its context")
        project_area = await urban_db_api.get_context_territories(territory_id, project_id)
        texts = await risk_calculator.get_texts(project_area)
        if len(texts['texts']) == 0:
            logger.info(f"No texts for this area")
            response = {}
            return response
        logger.info(f"Retrieving potential areas of coverage for project {project_id}")
        region_territories = await urban_db_api.get_territories(territory_id)
        logger.info("Calculating coverage")
        urban_areas = await risk_calculator.get_areas(region_territories, texts['texts'])
        logger.info(f"Generating links from project {project_id} to coverage areas")
        links = await risk_calculator.get_links(project_id, urban_areas, region_territories)
        urban_areas.drop(columns=['admin_center', 'is_city'], inplace=True)
        response = {
        'coverage_areas': json.loads(urban_areas.to_json()),
        'links_to_project': json.loads(links.to_json())
        }
        return response

    async def collect_texts(self, territory_id, project_id):
        logger.info(f"Retrieving texts for project {project_id} and its context")
        project_area = await urban_db_api.get_context_territories(territory_id, project_id)
        texts = await risk_calculator.get_texts(project_area)
        if len(texts['texts']) == 0:
            logger.info(f"No texts for this area")
            response = {}
            return response
        texts_df = texts['texts'].copy()
        texts_df = texts_df[['text', 'services', 'indicators']]
        texts_df = texts_df.groupby(
            ['text', 'services'] 
        ).agg({ 
            'indicators': lambda x: ', '.join(set(x))
        }).reset_index()  
        response = {
            'texts': json.loads(texts_df.to_json()),
        }
        return response
        

risk_calculator = RiskCalculation()