import pandas as pd
import geopandas as gpd

from shapely.geometry import shape, LineString
from geojson_pydantic import MultiPolygon, Polygon, Point
from app.risk_calculation.logic.constants import TEXTS

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
        local_crs = urban_areas.estimate_utm_crs()
        urban_areas['area'] = urban_areas.to_crs(local_crs).area
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