import geopandas as gpd
import pandas as pd
import json
from loguru import logger
from sqlalchemy import select
from geoalchemy2.shape import to_shape
from geoalchemy2.functions import ST_Intersects, ST_GeomFromText

from app.common.api.urbandb_api_gateway import urban_db_api
from app.common.db.database import Message, Emotion, Indicator, Service, MessageIndicator, MessageService, Territory, Group, GroupTerritory, database

class TextProcessing:
    async def get_texts(self, territory_gdf: gpd.GeoDataFrame):
        messages_crs = 4326  # TODO: может ли возникнуть проблема с хардкодом в дальнейшем?
        territory_gdf = territory_gdf.to_crs(epsg=messages_crs)
        territory_wkt = territory_gdf.unary_union.wkt

        async with database.session() as session:
            request = (
                select(
                    Message,
                    Emotion.name.label("emotion"),
                    Emotion.emotion_weight,
                    Indicator.name.label("indicator"),
                    Service.name.label("service"),
                    Territory.name.label("territory_name")
                )
                .select_from(Message)
                .join(Emotion, Message.emotion_id == Emotion.emotion_id)
                .join(MessageIndicator, Message.message_id == MessageIndicator.message_id)
                .join(Indicator, MessageIndicator.indicator_id == Indicator.indicator_id)
                .join(MessageService, Message.message_id == MessageService.message_id)
                .join(Service, MessageService.service_id == Service.service_id)
                .join(Group, Message.group_id == Group.group_id)
                .join(GroupTerritory, Group.group_id == GroupTerritory.group_id)
                .join(Territory, GroupTerritory.territory_id == Territory.territory_id)
                .where(
                    Message.type != "post",  # TODO: стоит поглядеть, как посты влияют на результат
                    ST_Intersects(
                        Message.geometry,
                        ST_GeomFromText(territory_wkt, messages_crs)
                    )
                )
            )
            result = await session.execute(request)
            rows = result.all()

        data = []
        for row in rows:
            message, emotion, emotion_weight, indicator, service, territory_name = row
            record = {
                "message_id": message.message_id,
                "text": message.text,
                "date": message.date,
                "views": message.views,
                "likes": message.likes,
                "reposts": message.reposts,
                "type": message.type,
                "location": message.location,
                "geometry": to_shape(message.geometry),
                "emotion": emotion,
                "emotion_weight": emotion_weight,
                "indicators": indicator,
                "services": service,
                "territory_name": territory_name,
            }
            data.append(record)

        gdf = gpd.GeoDataFrame(data, geometry="geometry", crs=messages_crs)
        return gdf

    async def collect_texts(self, territory_id, project_id):
        logger.info(f"Retrieving texts for project {project_id} and its context")
        project_area = await urban_db_api.get_context_territories(territory_id, project_id)
        texts = await text_processing.get_texts(project_area)
        texts["date"] = pd.to_datetime(texts["date"])
        texts["date"] = texts["date"].dt.strftime("%Y-%m-%d")
        texts = texts[['message_id', 'date', 'services']].sort_values(by='date').reset_index(drop=True)
        if len(texts) == 0:
            logger.info(f"No texts for this area")
            response = {}
            return response
        response = {
            'texts': texts.to_dict(),
        }
        return response
    
text_processing = TextProcessing()