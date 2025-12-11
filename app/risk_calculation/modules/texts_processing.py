import re
import geopandas as gpd
import pandas as pd
import numpy as np
from loguru import logger
from sqlalchemy import select, outerjoin
from geoalchemy2.shape import to_shape
from geoalchemy2.functions import ST_Intersects, ST_GeomFromText

from app.common.api.urbandb_api_gateway import urban_db_api
from app.common.modules.constants import CONSTANTS
from app.common.db.database import Message, Emotion, Indicator, Service, MessageIndicator, MessageService, Territory, Group, GroupTerritory
from app.common.db.db_engine import database
from app.common.exceptions.http_exception_wrapper import http_exception

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
                .outerjoin(Group, Message.group_id == Group.group_id)
                .outerjoin(
                    GroupTerritory,
                    Group.group_id == GroupTerritory.group_id
                )
                .outerjoin(
                    Territory,
                    GroupTerritory.territory_id == Territory.territory_id
                )
                .where(
                    Message.type != "post",
                    Message.geometry.is_not(None),
                    ST_Intersects(
                        Message.geometry,
                        ST_GeomFromText(territory_wkt, messages_crs)
                    )
                )
            )
            result = await session.execute(request)
            rows = result.all()

        if not rows:
            raise http_exception(
                status_code=404,
                msg="No messages in project territory",
                input_data={"territory_wkt": territory_wkt},
                detail="No messages for these conditions."
            )
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
        gdf['text'] = gdf['text'].map(self._clean_text)
        return gdf
    
    @staticmethod
    async def summarize_risk(df):
        score_df = df.groupby(['services', 'indicators'])['score'].mean().unstack(fill_value=0)
        score_df['total_score'] = score_df.sum(axis=1)
        score_df['total_score'] = (score_df['total_score'] / 5).clip(lower=0, upper=1)
        return score_df

    async def collect_texts(self, territory_id, project_id, period, token):
        """
        Retrieves texts for a given project and groups them by a specified period.
        
        Parameters:
            territory_id: идентификатор территории
            project_id: идентификатор проекта
            period: строка, определяющая период группировки ('day', 'week', 'month', 'year')
        
        Returns:
            Словарь, содержащий DataFrame с колонками:
                - category: категория сервисов
                - date: начало периода (в формате YYYY-MM-DD)
                - count: количество сообщений в этом периоде для данной категории
            Если между периодами имеются промежутки, они заполняются значением 0 в count.
        """
        logger.info(f"Retrieving texts for project {project_id} and its context")
        project_area = await urban_db_api.get_context_territories(territory_id, project_id, token)
        texts = await text_processing.get_texts(project_area)
        
        services_categories = CONSTANTS.json['services_categories']
        texts['category'] = texts['services'].map(services_categories) + ' инфраструктура'
        missing_services = texts[texts['category'].isna()].services.unique().tolist()
        if missing_services:
            logger.info(f"Attention: services not mapped to categories: {missing_services}")
        
        texts = texts.replace({np.nan: None})
        texts["date"] = pd.to_datetime(texts["date"])

        if period == 'day':
            texts['period'] = texts['date'].dt.floor('D')
            freq = 'D'
        elif period == 'week':
            texts['period'] = texts['date'].dt.to_period('W').apply(lambda r: r.start_time)
            freq = 'W-MON'
        elif period == 'month':
            texts['period'] = texts['date'].dt.to_period('M').apply(lambda r: r.start_time)
            freq = 'MS' 
        elif period == 'year':
            texts['period'] = texts['date'].dt.to_period('Y').apply(lambda r: r.start_time)
            freq = 'AS'  

        grouped = texts.groupby(['category', 'period']).size().reset_index(name='count')
        
        complete_dfs = []
        for cat in grouped['category'].unique():
            df_cat = grouped[grouped['category'] == cat].copy()
            start_date = df_cat['period'].min()
            end_date = df_cat['period'].max()
            full_dates = pd.DataFrame({'period': pd.date_range(start=start_date, end=end_date, freq=freq)})
            full_dates['category'] = cat
            df_cat = full_dates.merge(df_cat, on=['category', 'period'], how='left')
            df_cat['count'] = df_cat['count'].fillna(0).astype(int)
            complete_dfs.append(df_cat)
        
        if complete_dfs:
            result = pd.concat(complete_dfs)
            result.rename(columns={'period': 'date'}, inplace=True)
            result['date'] = result['date'].dt.strftime('%Y-%m-%d')
            return {'texts': result.to_dict(orient='records')}
        else:
            logger.info("No texts for this area")
            return {}
        
    @staticmethod
    def _clean_text(text):
        "Cleans texts from personal data in brackets and hyperlinks"
        RE_BRACKETS = re.compile(r'(?mi)^\s*(\[[^\]\n]*\]\s*,?\s*)+')
        RE_LINKS = re.compile(r'(?i)https?://\S+|h\s*t\s*t\s*p\s*s?\s*:\s*/\s*/[^\n"\)\]]+')
        text = RE_BRACKETS.sub('', text)
        text = RE_LINKS.sub('', text)
        text = re.sub(r'\s{2,}', ' ', text).strip(' \t-–,;')
        return text
    
text_processing = TextProcessing()