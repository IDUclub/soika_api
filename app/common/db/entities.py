from geoalchemy2.types import Geometry
from sqlalchemy import (
    MetaData,
    Column,
    TIMESTAMP,
    Enum,
    Integer,
    String,
    Table,
    text,
)


obj_metadata = MetaData()


class SoikaTextSourseTypeEnum(Enum):
    VK = "VK"
    MSPB= "MSPB"


class SoikaTextTypeEnum(Enum):
    POST = "POST"
    COMMENT = "COMMENT"
    REPLY = "REPLY"
    COMPLAINT = "COMPLAINT"


cities = Table(
    "texts",
    obj_metadata,
    Column("text", String(500), nullable=False),
        Column("vkid", Integer, nullable=False),
        Column("public", String(64), nullable=False),
        Column("parent_id", Integer, nullable=False),
        Column("location", String(100), nullable=True, default=None),
        Column("text_type", Enum(SoikaTextSourseTypeEnum), nullable=False),
        Column("text_source", Enum(SoikaTextTypeEnum), nullable=True),
        Column("date", TIMESTAMP(timezone=True), server_default=text("now()"), nullable=False),
        Column("city_function", String(64)),
        Column("topic", Integer, nullable=False),
        Column('emotion', String(64), nullable=True),
        Column("service_type", String(64), nullable=True),
        Column("demand_type", String(64), nullable=True),
        Column(
            "center",
            Geometry(
                "POINT", srid=4326, spatial_index=False, from_text="ST_GeomFromEWKT", name="center")
            ),
            nullable=True,
            default=None,
        )
