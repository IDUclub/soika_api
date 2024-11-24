import re
import nltk
import pandas as pd
import geopandas as gpd
from sloyka import AreaMatcher, TextClassifiers, Geocoder
from shapely.geometry import shape, LineString
from geojson_pydantic import MultiPolygon, Polygon, Point
from app.risk_calculation.logic.constants import TEXTS
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

    def run_geocoding_and_classification(self, area_matcher, emotion_classifier, tokenizer, place_name):
        """Geocoding and emotion classification"""
        self.source_data_processed = area_matcher.run(self.source_data_processed, place_name=place_name)
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

class RiskCalculation:
    def __init__(self, emotion_weights=None):
        """Initialize the class with emotion weights."""
        # Set emotion weights. Custom weights can be passed via the `emotion_weights` parameter.
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

    async def calculate_score(self, dataframe):
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

    async def score_table(self, dataframe):
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
    ) -> pd.DataFrame:
        """
        Retrieves the source texts in the given territory.

        Args:
            territory_gdf (gpd.GeoDataFrame): A GeoDataFrame representing the area of interest.

        Returns:
            DataFrame:
        """
        texts = gpd.clip(TEXTS.gdf, territory_gdf)
        return texts

    @staticmethod
    async def get_areas(urban_areas: gpd.GeoDataFrame, texts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        urban_areas = urban_areas.merge(texts['best_match'].value_counts().rename('count'), left_on='name', right_index=True, how='left')
        urban_areas = urban_areas[['name', 'geometry', 'count']]
        urban_areas.dropna(subset='count', inplace=True)
        urban_areas['area'] = urban_areas.to_crs(3857).area
        urban_areas = urban_areas.sort_values(by='area', ascending=False).drop_duplicates(subset='name', keep='first')
        urban_areas.drop(columns=['area'], inplace=True)
        return urban_areas

    @staticmethod
    async def get_links(project_territory: gpd.GeoDataFrame, urban_areas: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        project_centroid = project_territory.geometry.unary_union.centroid
        lines_data = []
        for _, area in urban_areas.iterrows():
            area_centroid = area.geometry.centroid
            line = LineString([area_centroid, project_centroid])
            lines_data.append({"urban_area": area["name"], "geometry": line})
        lines_gdf = gpd.GeoDataFrame(lines_data, geometry="geometry", crs=project_territory.crs)
        return lines_gdf

risk_calculator = RiskCalculation()