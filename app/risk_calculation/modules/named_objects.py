import geopandas as gpd
import json
import ast
from loguru import logger
from app.common.api.urbandb_api_gateway import urban_db_api


from sqlalchemy import select
from geoalchemy2.shape import to_shape
from geoalchemy2.functions import ST_Intersects, ST_GeomFromText

from app.common.api.urbandb_api_gateway import urban_db_api
from app.common.db.database import Message, NamedObject, MessageNamedObject
from app.common.db.db_engine import database
from app.risk_calculation.modules.texts_processing import text_processing

class NamedObjects:
    @staticmethod        
    async def get_objects(territory_gdf: gpd.GeoDataFrame):
        """
        Извлекает именованные объекты для заданной территории.
        """
        messages_crs = 4326  # TODO: проверить возможные проблемы с хардкодом
        territory_gdf = territory_gdf.to_crs(epsg=messages_crs)
        territory_wkt = territory_gdf.unary_union.wkt

        async with database.session() as session:
            query = (
                select(NamedObject)
                .where(
                    ST_Intersects(
                        NamedObject.geometry,
                        ST_GeomFromText(territory_wkt, messages_crs)
                    )
                )
            )
            result = await session.execute(query)
            objects = result.scalars().all()

            data = []
            for obj in objects:
                mapping_query = (
                    select(Message.text)
                    .join(MessageNamedObject, Message.message_id == MessageNamedObject.message_id)
                    .where(MessageNamedObject.named_object_id == obj.named_object_id)
                )
                mapping_result = await session.execute(mapping_query)
                texts = mapping_result.scalars().all()
                if isinstance(texts, str):
                    texts = [texts]
                elif texts is None:
                    texts = []

                estimated_location_list = (
                    obj.estimated_location.split("; ") if obj.estimated_location else []
                )
                object_description_list = (
                    obj.object_description.split("; ") if obj.object_description else []
                )
                
                object_description_list = [text_processing.clean_text(x) for x in object_description_list if x]
                estimated_location_list = [text_processing.clean_text(x) for x in estimated_location_list if x]
                if (
                    isinstance(object_description_list, list)
                    and len(object_description_list) > 1
                    and all(isinstance(x, str) and len(x) <= 1 for x in object_description_list)
                ):
                    object_description_list = ["".join((" " if x == "" else x) for x in object_description_list)]
                # TODO: костыль. Поменять механизм загрузки в базу osm_id
                if obj.osm_id == 0:
                    osm_id = None
                else:
                    osm_id = obj.osm_id
                # TODO: костыль. Надо будет потом убедиться. что при поиске осм объектов не может быть больше одного совпадения
                if obj.osm_tag == "":
                    osm_tag = None
                else:
                    try:
                        osm_tag = ast.literal_eval(obj.osm_tag)
                        if not isinstance(osm_tag, list):
                            osm_tag = [osm_tag]
                    except Exception:
                        osm_tag = [obj.osm_tag]
                texts = [text_processing.clean_text(text) for text in texts]
                if not texts or all((t is None) or (len(t) == 0) for t in texts):
                    continue
                record = {
                    "named_object_id": obj.named_object_id,
                    "object_name": obj.object_name,
                    "estimated_location": estimated_location_list,
                    "object_description": object_description_list,
                    "osm_id": osm_id,
                    "accurate_location": obj.accurate_location,
                    "osm_tag": osm_tag,
                    "count": obj.count,
                    "text": texts,
                    "geometry": to_shape(obj.geometry),
                }
                data.append(record)

        gdf = gpd.GeoDataFrame(data, geometry="geometry", crs=messages_crs)
        return gdf

    async def collect_named_objects(self, territory_id, project_id, token):
        logger.info(f"Retrieving objects for project {project_id} and its context")
        project_area = await urban_db_api.get_context_territories(territory_id, project_id, token)
        local_objects = await named_objects_collection.get_objects(project_area)

        if local_objects.empty:
            logger.info("No objects for this area")
            return {}

        response = {
            'named_objects': json.loads(local_objects.to_json())
        }
        return response
    
named_objects_collection = NamedObjects()