# import asyncio
# import geopandas as gpd
# import pandas as pd
# from shapely.geometry import shape
# from shapely import wkt
# from typing import List
# from geojson_pydantic import MultiPolygon, Polygon, Point
# from blocksnet import Provision, City, Accessibility, ACCESSIBILITY_TO_COLUMN

# from app.llm_tables.dto.llm_impact_evaluation_dto import ImpactEvalDTO
# from app.llm_tables.logic.constants import CITY, TEXTS, DEMOLITION_TEXTS
# from app.common.exceptions.http_exception_wrapper import http_exception
# from app.llm_tables.services import DistrictsService, BlocksService, CitiesService, ServicesService, BuildingsService, \
#     MunicipalitiesService

import re
import nltk
import pandas as pd
import geopandas as gpd
from soyka import AreaMatcher, TextClassifiers, Geocoder
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm

nltk.download('punkt')
nltk.download('wordnet')


class DataStructurer:
    def __init__(self, source_data_path, indicators_path, services_path):
        """Initialization of Structurer"""
        self.source_data = gpd.read_file(source_data_path)
        self.indicators = pd.read_excel(indicators_path)
        self.services = pd.read_excel(services_path)
        
        self.services_dict = {
            row['service']: row['keywords'].split(', ')
            for _, row in self.services.iterrows()
        }
        self.indicators_dict = {
            row['indicators']: row['keywords'].split(', ')
            for _, row in self.indicators.iterrows()
        }
        
        self.lemmatizer = WordNetLemmatizer()

    def lemmatize_text(self, text):
        """Lemmatization of texts"""
        return [self.lemmatizer.lemmatize(word.lower()) for word in word_tokenize(text)]

    def lemmatize_keywords(self, keywords):
        """Lemmatization of keywords"""
        lemmatized_phrases = []
        for phrase in keywords:
            lemmatized_phrases.append(
                " ".join(self.lemmatizer.lemmatize(word.lower()) for word in word_tokenize(phrase))
            )
        return lemmatized_phrases

    def find_services_in_text(self, text, keywords_dict):
        """Search for services or indicators in the text"""
        found_items = []
        lemmatized_text = " ".join(self.lemmatize_text(text))
        
        for service, keywords in keywords_dict.items():
            lemmatized_keywords = self.lemmatize_keywords(keywords)
            if any(re.search(rf'\b{re.escape(keyword)}\b', lemmatized_text) for keyword in lemmatized_keywords):
                found_items.append(service)
        
        return ', '.join(found_items) if found_items else None

    def process_source_data(self):
        """Structurization of source data"""
        tqdm.pandas()
        self.source_data['views.count'].fillna(0, inplace=True)
        self.source_data['reposts.count'].fillna(0, inplace=True)
        
        self.source_data['services'] = self.source_data.text.progress_map(
            lambda x: self.find_services_in_text(x, self.services_dict)
        )
        self.source_data['indicators'] = self.source_data.text.progress_map(
            lambda x: self.find_services_in_text(x, self.indicators_dict)
        )
        
        self.source_data_processed = self.source_data[
            self.source_data.services.notna() & self.source_data.indicators.notna()
        ]

    def run_geocoding_and_classification(self, area_matcher, emotion_classifier, tokenizer, osm_id):
        """Geocoding and emotion classification"""
        self.source_data_processed = area_matcher.run(self.source_data_processed, osm_id=osm_id)
        df_areas = area_matcher.get_osm_areas(osm_id)

        self.source_data_processed['text'] = self.source_data_processed.text.progress_map(
            lambda x: truncate_sentence(x, tokenizer)
        )
        emotions = self.source_data_processed.text.progress_map(
            lambda x: emotion_classifier.run_text_classifier(x)
        )
        self.source_data_processed = pd.concat(
            [
                self.source_data_processed.reset_index(drop=True),
                pd.DataFrame(emotions.tolist(), columns=['emotion', 'emotion_prob'])
            ],
            axis=1
        )

    def get_processed_data(self):
        """Возвращает обработанные данные."""
        return self.source_data_processed


# Пример использования класса
# if __name__ == "__main__":
#     processor = DataProcessor(
#         source_data_path='source_data_processed.geojson',
#         indicators_path='F:/Coding/fn34/indicators_keywords.xlsx',
#         services_path='F:/Coding/fn34/services_keywords.xlsx'
#     )
#     processor.process_source_data()

#     # Инициализация внешних классов
#     area_matcher = AreaMatcher()
#     emotion_classifier = TextClassifiers("Sandrro/emotions_classificator")

#     # Запуск геокодирования и классификации
#     processor.run_geocoding_and_classification(area_matcher, emotion_classifier, tokenizer, osm_id=176095)

#     # Получение результата
#     processed_data = processor.get_processed_data()
#     print(processed_data)

class RiskCalculation:
    def __init__(self, emotion_weights=None):
        """Initialize the class with emotion weights."""
        # Set emotion weights. Custom weights can be passed via the `emotion_weights` parameter.
        self.emotion_weights = emotion_weights or {'positive': 1.5, 'neutral': 1, 'negative': 1.5}

    def expand_rows_by_columns(self, dataframe, columns):
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

    def calculate_score(self, dataframe):
        """
        Calculates scores based on emotions, views, likes, and reposts.
        
        Args:
            dataframe (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with an added `score` column.
        """
        df = dataframe.copy()
        df['emotion_weight'] = df['emotion'].map(self.emotion_weights)

        # Normalize columns
        df['minmaxed_views'] = (df['views.count'] - df['views.count'].min()) / (df['views.count'].max() - df['views.count'].min())
        df['minmaxed_likes'] = (df['likes.count'] - df['likes.count'].min()) / (df['likes.count'].max() - df['likes.count'].min())
        df['minmaxed_reposts'] = (df['reposts.count'] - df['reposts.count'].min()) / (df['reposts.count'].max() - df['reposts.count'].min())

        # Calculate the final `score`
        df['score'] = df.apply(
            lambda row: (row['minmaxed_views'] + row['minmaxed_likes'] + row['minmaxed_reposts'] + 1) * row['emotion_weight'],
            axis=1
        )
        df['score'] = df['score'].round(4)

        # Remove temporary columns
        return df.drop(columns=['minmaxed_views', 'minmaxed_likes', 'minmaxed_reposts'])

    def score_table(self, dataframe):
        """
        Generates a table with average scores for each (service, indicator) pair.
        
        Args:
            dataframe (pd.DataFrame): Input DataFrame.

        Returns:
            dict: A dictionary where keys are indicators and values are scores for each service.
        """
        grouped = dataframe.groupby(['services', 'indicators'])['score'].mean().unstack(fill_value=0)
        grouped = grouped.round(4)
        score_dict = grouped.to_dict()

        return score_dict
