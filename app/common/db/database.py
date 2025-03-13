from sqlalchemy import (
    URL,
    NullPool,
    Column,
    Integer,
    String,
    Boolean,
    Float,
    DateTime,
    ForeignKey,
)
from geoalchemy2.types import Geometry
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import Executable

from app.common.config import config


Base = declarative_base()


class DatabaseEngine:
    def __init__(self):
        self.url = URL.create(
            config.get("DB_ENGINE"),
            username=config.get("DB_USERNAME"),
            password=config.get("DB_PASSWORD"),
            host=config.get("DB_HOST"),
            port=config.get("DB_PORT"),
            database=config.get("DB_DATABASE"),
        )
        self.engine = create_async_engine(
            self.url,
            echo=True,
            poolclass=NullPool,
        )
        self.conn = self.engine.connect()
        self.session = sessionmaker(
            self.engine, expire_on_commit=False, class_=AsyncSession
        )

    async def execute_query(self, query: Executable):
        async with self.session() as session:
            return await session.execute(query)


class Territory(Base):
    __tablename__ = "territory"
    territory_id = Column(Integer, primary_key=True, nullable=False)
    name = Column(String)
    matched_territory = Column(String)


class Group(Base):
    __tablename__ = "group"
    group_id = Column(Integer, primary_key=True, nullable=False)
    name = Column(String)
    group_domain = Column(String)
    matched_territory = Column(String)

class GroupTerritory(Base):
    __tablename__ = "group_teritory"
    group_id = Column(
        Integer, ForeignKey("group.group_id"), primary_key=True, nullable=False
    )
    territory_id = Column(
        Integer, ForeignKey("territory.territory_id"), primary_key=True, nullable=False
    ) 

class Message(Base):
    __tablename__ = "message"
    message_id = Column(Integer, primary_key=True, nullable=False)
    text = Column(String)
    date = Column(DateTime(timezone=True))
    views = Column(Integer)
    likes = Column(Integer)
    reposts = Column(Integer)
    type = Column(String)
    parent_message_id = Column(Integer)
    group_id = Column(Integer)
    emotion_id = Column(Integer)
    score = Column(Float)
    geometry = Column(
        "geometry",
        Geometry(
            spatial_index=False,
            from_text="ST_GeomFromEWKT",
            name="geometry",
            nullable=False,
        ),
        nullable=False,
    )
    location = Column(String)
    is_processed = Column(Boolean)


class NamedObject(Base):
    __tablename__ = "named_object"
    named_object_id = Column(Integer, primary_key=True, nullable=False)
    estimated_location = Column(String)
    object_description = Column(String)
    osm_id = Column(Integer)
    accurate_location = Column(String)
    count = Column(Integer)
    text_id = Column(Integer)
    osm_tag = Column(String)
    geometry = Column(
        "geometry",
        Geometry(
            spatial_index=False,
            from_text="ST_GeomFromEWKT",
            name="geometry",
            nullable=False,
        ),
        nullable=False,
    )
    is_processed = Column(Boolean)


class Emotion(Base):
    __tablename__ = "emotion"
    emotion_id = Column(Integer, primary_key=True, nullable=False)
    name = Column(String)
    emotion_weight = Column(Float)


class Indicator(Base):
    __tablename__ = "indicator"
    indicator_id = Column(Integer, primary_key=True, nullable=False)
    name = Column(String)


class MessageIndicator(Base):
    __tablename__ = "message_indicator"
    message_id = Column(
        Integer, ForeignKey("message.message_id"), primary_key=True, nullable=False
    )
    indicator_id = Column(
        Integer, ForeignKey("indicator.indicator_id"), primary_key=True, nullable=False
    )


class Service(Base):
    __tablename__ = "service"
    service_id = Column(Integer, primary_key=True, nullable=False)
    name = Column(String)
    value_id = Column(Integer)


class MessageService(Base):
    __tablename__ = "message_service"
    message_id = Column(
        Integer, ForeignKey("message.message_id"), primary_key=True, nullable=False
    )
    service_id = Column(
        Integer, ForeignKey("service.service_id"), primary_key=True, nullable=False
    )


database = DatabaseEngine()
