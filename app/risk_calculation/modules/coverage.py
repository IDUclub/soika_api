import geopandas as gpd
import json
import asyncio

from loguru import logger
from shapely.geometry import LineString
from app.risk_calculation.modules.geo_utils import geo_utils
from app.risk_calculation.modules.texts_processing import text_processing
from app.common.api.urbandb_api_gateway import urban_db_api

class CoverageCalculation:
    @staticmethod
    async def get_areas(urban_areas: gpd.GeoDataFrame, texts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Определяет зоны охвата, используя данные об урбанизированных территориях и тексты.
        Здесь используется helper reproject_to_local для унифицированного вычисления площадей.
        """
        local_areas, local_crs = geo_utils.reproject_to_local(urban_areas)
        local_areas["area"] = local_areas.area

        urban_areas = urban_areas.merge(
            texts["territory_name"].value_counts().rename("count"),
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
    async def get_links(project_id: int, urban_areas: gpd.GeoDataFrame, region_territories: gpd.GeoDataFrame, token) -> gpd.GeoDataFrame:
        """
        Создает линейные связи между центроидом проекта и центроидами urban-территорий.
        """
        project_centroid = await urban_db_api.get_project_territory_centroid(project_id, token)
        areas = urban_areas.copy()
        areas["geometry"] = await asyncio.gather(
            *[CoverageCalculation.get_area_centroid(area, region_territories) for _, area in areas.iterrows()]
        )
        areas["geometry"] = areas.apply(
            lambda area: LineString([area.geometry, project_centroid]), axis=1
        )
        grouped = areas.groupby("geometry").agg({"name": lambda x: list(x)}).reset_index()
        grouped.rename(columns={"name": "urban_area"}, inplace=True)
        links_gdf = gpd.GeoDataFrame(grouped, geometry="geometry", crs=urban_areas.crs)
        return links_gdf

    async def calculate_coverage(self, territory_id, project_id, token):
        """
        Расчет охвата: получение текстов, определение urban-территорий и построение связей с проектом.
        """
        logger.info(f"Retrieving texts for project {project_id} and its context")
        project_area = await urban_db_api.get_context_territories(territory_id, project_id, token)
        texts = await text_processing.get_texts(project_area)
        if len(texts) == 0:
            logger.info("No texts for this area")
            return {}

        logger.info("Retrieving potential areas of coverage")
        region_territories = await urban_db_api.get_territories(territory_id, token)
        urban_areas = await CoverageCalculation.get_areas(region_territories, texts)
        logger.info("Generating links from project to coverage areas")
        links = await CoverageCalculation.get_links(project_id, urban_areas, region_territories, token)

        urban_areas.drop(columns=["admin_center", "is_city"], inplace=True)
        response = {
            "coverage_areas": json.loads(urban_areas.to_json()),
            "links_to_project": json.loads(links.to_json()),
        }
        return response
    
coverage_calculation = CoverageCalculation()