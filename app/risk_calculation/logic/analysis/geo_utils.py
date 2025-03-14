import geopandas as gpd

class GeoUtils:
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
        local_territory, local_crs = GeoUtils.reproject_to_local(territory_gdf)
        local_source = source_gdf.to_crs(local_crs)
        clipped = gpd.clip(local_source, local_territory)
        return clipped.to_crs(epsg=4326)
    
geo_utils = GeoUtils()