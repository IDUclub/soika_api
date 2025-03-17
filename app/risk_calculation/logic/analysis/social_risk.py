import pandas as pd

from loguru import logger
from app.risk_calculation.logic.analysis.texts_processing import text_processing
from app.common.api.urbandb_api_gateway import urban_db_api

class RiskCalculation:
    def __init__(self, emotion_weights=None):
        """Initialize the class with emotion weights."""
        self.emotion_weights = emotion_weights or {'positive': 1.5, 'neutral': 1, 'negative': 1.5}

    @staticmethod
    def minmax_normalize(series: pd.Series) -> pd.Series:
        """
        Выполняет нормализацию данных по методу min-max.

        Returns:
            pd.Series: Нормализованная серия
        """
        return (series - series.min()) / (series.max() - series.min())

    async def calculate_score(self, df):
        """
        Вычисляет итоговый скор на основе нормализованных показателей и веса эмоции.
        """
        df = df.copy()
        df["emotion_weight"] = df["emotion"].map(self.emotion_weights)
        df["minmaxed_views"] = risk_calculation.minmax_normalize(df["views"])
        df["minmaxed_likes"] = risk_calculation.minmax_normalize(df["likes"])
        df["minmaxed_reposts"] = risk_calculation.minmax_normalize(df["reposts"])
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
        score_df["top_indicators"] = score_df_numeric.apply(risk_calculation.get_top_indicators, axis=1)
        score_df['risk_rating'] = score_df_numeric.sum(axis=1).clip(upper=5).round(0).astype(int)
        score_df['risk_level'] = score_df['risk_rating'].map(lambda x: "высокой" if x >= 4 else ("средней" if 2 <= x < 4 else "низкой"))
    
        emotion_table = df.groupby('services')['emotion'].agg(lambda x: x.mode().iloc[0])
        activity_table = df.groupby('services')['activity_level'].agg(lambda x: x.mode().iloc[0])
        
        final_table = score_df.join(emotion_table)
        final_table = final_table.join(activity_table)
        final_table["description"] = final_table.apply(risk_calculation.generate_description, axis=1)
        final_table = final_table[['risk_rating', 'description']]
        return final_table

    async def calculate_social_risk(self, territory_id, project_id):
        logger.info(f"Retrieving texts for project {project_id} and its context")
        project_area = await urban_db_api.get_context_territories(territory_id, project_id)
        texts = await text_processing.get_texts(project_area)

        if len(texts) == 0:
            logger.info(f"No texts for this area")
            response = {}
            return response

        logger.info(f"Calculating social risk for project {project_id} and its context")
        scored_texts = await risk_calculation.calculate_score(texts)
        score_df = await risk_calculation.score_table(scored_texts)

        texts_df = texts.copy()
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
    
risk_calculation = RiskCalculation()