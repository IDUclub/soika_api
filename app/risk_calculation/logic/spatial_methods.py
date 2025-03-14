import pandas as pd
import geopandas as gpd
import json
import asyncio

from loguru import logger
from shapely.geometry import shape, LineString
from geojson_pydantic import MultiPolygon, Polygon, Point
from app.risk_calculation.logic.constants import TEXTS, CONSTANTS, OBJECTS
from app.common.api.urbandb_api_gateway import urban_db_api
from app.common.api.values_api_gateway import values_api

class RiskCalculation:
    def __init__(self, emotion_weights=None):
        """Initialize the class with emotion weights."""
        self.emotion_weights = emotion_weights or {'positive': 1.5, 'neutral': 1, 'negative': 1.5}

    @staticmethod
    def reproject_to_local(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, any]:
        """
        Приводит GeoDataFrame к локальной UTM проекции.

        Returns:
            Tuple: (GeoDataFrame в локальной проекции, локальная CRS)
        """
        local_crs = gdf.estimate_utm_crs()
        local_gdf = gdf.to_crs(local_crs)
        return local_gdf, local_crs

    @staticmethod
    def clip_and_reproject(source_gdf: gpd.GeoDataFrame, territory_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Приводит оба GeoDataFrame (исходный и территории) к одной локальной проекции,
        обрезает исходный по территории и возвращает результат в системе EPSG:4326.
        """
        local_territory, local_crs = risk_calculator.reproject_to_local(territory_gdf)
        local_source = source_gdf.to_crs(local_crs)
        clipped = gpd.clip(local_source, local_territory)
        return clipped.to_crs(epsg=4326)

    @staticmethod
    def minmax_normalize(series: pd.Series) -> pd.Series:
        """
        Выполняет нормализацию данных по методу min-max.

        Returns:
            pd.Series: Нормализованная серия
        """
        return (series - series.min()) / (series.max() - series.min())
    
    async def expand_rows_by_columns(self, dataframe, columns):
        """
        Расширяет строки DataFrame по заданным столбцам с разделёнными значениями.
        """
        expanded_df = dataframe.copy()
        for column in columns:
            expanded_df[column] = expanded_df[column].str.split(", ")
            expanded_df = expanded_df.explode(column, ignore_index=True)
        return expanded_df

    @staticmethod
    async def to_gdf(geometry: Polygon | Point | MultiPolygon) -> gpd.GeoDataFrame:
        """
        Преобразует переданную геометрию в GeoDataFrame.
        """
        if isinstance(geometry, Point):
            gs: gpd.GeoSeries = gpd.GeoSeries([geometry], crs=4326).to_crs(3857)
            buffer = gs.buffer(500)
            gdf = gpd.GeoDataFrame(geometry=buffer, crs=3857)
            gdf.to_crs(4326, inplace=True)
        else:
            geometry = {"geometry": [shape(geometry)]}
            gdf = gpd.GeoDataFrame(geometry, geometry="geometry", crs=4326)
        return gdf

    @staticmethod
    async def get_texts(territory_gdf: gpd.GeoDataFrame):
        """
        Извлекает тексты, относящиеся к заданной территории.
        Применяется общая функция для обрезки и приведения CRS.
        """
        texts_gdf = TEXTS.gdf.copy()
        texts_gdf = texts_gdf[texts_gdf["type"] != "post"] # TODO: исключение постов стоит потом пересмотреть
        local_texts = risk_calculator.clip_and_reproject(texts_gdf, territory_gdf)
        return {"texts": local_texts}

    async def calculate_score(self, df):
        """
        Вычисляет итоговый скор на основе нормализованных показателей и веса эмоции.
        """
        df = df.copy()
        df["emotion_weight"] = df["emotion"].map(self.emotion_weights)
        df["minmaxed_views"] = risk_calculator.minmax_normalize(df["views.count"])
        df["minmaxed_likes"] = risk_calculator.minmax_normalize(df["likes.count"])
        df["minmaxed_reposts"] = risk_calculator.minmax_normalize(df["reposts.count"])
        df["engagement_score"] = df.fillna(0).apply(
            lambda row: row["minmaxed_views"] +
                        row["minmaxed_likes"] +
                        row["minmaxed_reposts"] + 1, axis=1
        )
        low_threshold = df["engagement_score"].quantile(1 / 3)
        high_threshold = df["engagement_score"].quantile(2 / 3)
        df["activity_level"] = df["engagement_score"].apply(
            lambda x: "активно" if x >= high_threshold else ("умеренно" if x >= low_threshold else "мало")
        )
        df["score"] = (df["engagement_score"] * df["emotion_weight"]).round(4)
        return df.drop(
            columns=["minmaxed_views", "minmaxed_likes", "minmaxed_reposts", "engagement_score"]
        )

    @staticmethod
    def get_top_indicators(row):
        nonzero_values = row[row != 0]
        top_indicators = nonzero_values.nlargest(4).index.tolist()
        declensions = {
            "Строительство": "строительство",
            "Снос": "снос",
            "Противоречие": "противоречие",
            "Доступность": "доступность",
            "Обеспеченность": "обеспеченность"
        }
        declined = [declensions.get(ind, ind) for ind in top_indicators]
        return f"{', '.join(declined)}" if declined else "без явных индикаторов"

    @staticmethod
    def generate_description(row):
        service_name = row.name
        risk_level = row["risk_level"]
        top_indicators = row["top_indicators"]
        activity_level = row["activity_level"]
        emotion = row["emotion"]

        risk_text = f"Сервис «{service_name}» характеризуется {risk_level} степенью общественного резонанса."
        indicators_list = top_indicators.split(', ')
        indicator_count = len(indicators_list)
        number_words = {
            1: "один",
            2: "два",
            3: "три",
            4: "четыре",
            5: "пять"
        }
        
        indicator_count_word = number_words.get(indicator_count, str(indicator_count))

        if indicator_count == 1:
            indicators_text = (
                f"Среди показателей оценки уровня общественного резонанса выделяется один ключевой - {indicators_list[0]} сервиса данного типа."
            )
        elif indicator_count == 2:
            indicators_text = (
                f"Среди показателей оценки уровня общественного резонанса выделяется {indicator_count_word}: "
                f"{' и '.join(indicators_list)} сервисов данного типа."
            )
        else:
            indicators_text = (
                f"Среди показателей оценки уровня общественного резонанса выделяется {indicator_count_word}: "
                f"{', '.join(indicators_list)} сервиса данного типа."
            )
        activity_text = (
            f"Сервис {activity_level} обсуждается пользователями."
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
        if activity_level == "активно":
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
        score_df['risk_level'] = score_df['risk_rating'].map(lambda x: "высокой" if x >= 4 else ("средней" if 2 <= x < 4 else "низкой"))
    
        emotion_table = df.groupby('services')['emotion'].agg(lambda x: x.mode().iloc[0])
        activity_table = df.groupby('services')['activity_level'].agg(lambda x: x.mode().iloc[0])
        
        final_table = score_df.join(emotion_table)
        final_table = final_table.join(activity_table)
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
        tasks = [risk_calculator.get_level_4_territory(territory_id) for territory_id in territories_to_fetch['territory_id']]
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

    def generate_category_table(self):
        mapping_df = pd.DataFrame(CONSTANTS.json['service_to_values_mapping'])
        mapping_df['category'] = mapping_df['values'] + '/' + mapping_df['social_group']
        mapping_df['category'] =  mapping_df['category'].map(risk_calculator.generate_value_names)
        category_table = mapping_df.groupby('category').agg({
            'services': lambda x: ', '.join(x.dropna().astype(str))
        })
        return category_table

    async def calculate_values_to_risk_data(self, territory_id, project_id):
        logger.info(f"Retrieving texts for project {project_id} and its context")
        project_area = await urban_db_api.get_context_territories(territory_id, project_id)
        texts = await risk_calculator.get_texts(project_area)

        if len(texts['texts']) == 0:
            logger.info(f"No texts for this area")
            response = {}
            return response

        logger.info(f"Calculating risk-to-value table for project {project_id} and its context")
        scored_texts = await risk_calculator.calculate_score(texts['texts'])
        score_df = await risk_calculator.summarize_risk(scored_texts)
        risk_series = await risk_calculator.map_risk_score(score_df)

        municipalities = await risk_calculator.fetch_municipalities(project_area)
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
        values_to_risk_table['category'] = values_to_risk_table['category'].map(risk_calculator.generate_value_names)
        values_to_risk_table.dropna(subset='Поддержка ценностей', inplace=True)
        category_table = risk_calculator.generate_category_table()
        values_to_risk_table = values_to_risk_table.merge(category_table, on='category', how='left')
        values_to_risk_table['services'] = values_to_risk_table['services'].fillna('Нет данных по сервисам')
        response = {'values_to_risk_table': values_to_risk_table.to_dict(orient='records')}
        logger.info(f"Table response generated")
        return response

    @staticmethod
    async def get_areas(urban_areas: gpd.GeoDataFrame, texts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Определяет зоны охвата, используя данные об урбанизированных территориях и тексты.
        Здесь используется helper reproject_to_local для унифицированного вычисления площадей.
        """
        local_areas, local_crs = risk_calculator.reproject_to_local(urban_areas)
        local_areas["area"] = local_areas.area

        urban_areas = urban_areas.merge(
            texts["best_match"].value_counts().rename("count"),
            left_on="name",
            right_index=True,
            how="left",
        )
        urban_areas = urban_areas[["name", "territory_id", "admin_center", "is_city", "geometry", "count"]]
        urban_areas.dropna(subset=["count"], inplace=True)

        local_areas = local_areas.loc[urban_areas.index].sort_values("area", ascending=False)
        urban_areas = local_areas.drop(columns=["area"]).to_crs(epsg=4326)
        return urban_areas

    @staticmethod
    async def get_area_centroid(area, region_territories):
        """
        Определяет центроид территории.
        Если это город – используется центроид объекта, иначе берется центроид административного центра.
        """
        if area["is_city"]:
            return area.geometry.centroid
        else:
            filtered_region = region_territories.loc[
                region_territories["territory_id"] == area.admin_center
            ]
            return filtered_region.geometry.centroid.iloc[0]

    @staticmethod
    async def get_links(project_id: int, urban_areas: gpd.GeoDataFrame, region_territories: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Создает линейные связи между центроидом проекта и центроидами urban-территорий.
        """
        project_centroid = await urban_db_api.get_project_territory_centroid(project_id)
        areas = urban_areas.copy()
        areas["geometry"] = await asyncio.gather(
            *[risk_calculator.get_area_centroid(area, region_territories) for _, area in areas.iterrows()]
        )
        areas["geometry"] = areas.apply(
            lambda area: LineString([area.geometry, project_centroid]), axis=1
        )
        grouped = areas.groupby("geometry").agg({"name": lambda x: list(x)}).reset_index()
        grouped.rename(columns={"name": "urban_area"}, inplace=True)
        links_gdf = gpd.GeoDataFrame(grouped, geometry="geometry", crs=urban_areas.crs)
        return links_gdf

    async def calculate_coverage(self, territory_id, project_id):
        """
        Расчет охвата: получение текстов, определение urban-территорий и построение связей с проектом.
        """
        logger.info(f"Retrieving texts for project {project_id} and its context")
        project_area = await urban_db_api.get_context_territories(territory_id, project_id)
        texts = await risk_calculator.get_texts(project_area)
        if len(texts["texts"]) == 0:
            logger.info("No texts for this area")
            return {}

        logger.info("Retrieving potential areas of coverage")
        region_territories = await urban_db_api.get_territories(territory_id)
        urban_areas = await risk_calculator.get_areas(region_territories, texts["texts"])
        logger.info("Generating links from project to coverage areas")
        links = await risk_calculator.get_links(project_id, urban_areas, region_territories)

        urban_areas.drop(columns=["admin_center", "is_city"], inplace=True)
        response = {
            "coverage_areas": json.loads(urban_areas.to_json()),
            "links_to_project": json.loads(links.to_json()),
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
        
    @staticmethod
    async def get_objects(territory_gdf: gpd.GeoDataFrame):
        """
        Извлекает объекты для заданной территории с использованием общей функции обрезки.
        """
        objects_gdf = OBJECTS.gdf.copy()
        local_objects = risk_calculator.clip_and_reproject(objects_gdf, territory_gdf)
        return {"objects": local_objects.to_crs(epsg=4326)}

    async def collect_named_objects(self, territory_id, project_id):
        logger.info(f"Retrieving objects for project {project_id} and its context")
        project_area = await urban_db_api.get_context_territories(territory_id, project_id)
        objects_result = await risk_calculator.get_objects(project_area)
        local_objects = objects_result['objects']

        if local_objects.empty:
            logger.info("No objects for this area")
            return {}

        response = {
            'named_objects': json.loads(local_objects.to_json())
        }
        return response


risk_calculator = RiskCalculation()