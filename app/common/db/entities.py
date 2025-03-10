from geoalchemy2.types import Geometry
from sqlalchemy import MetaData, Boolean, Text, SmallInteger
from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Table,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB


obj_metadata = MetaData()


class CityDivisionTypeEnum(Enum):
    ADMIN_UNIT_PARENT = "ADMIN_UNIT_PARENT"
    MUNICIPALITY_PARENT = "MUNICIPALITY_PARENT"
    NO_PARENT = "NO_PARENT"


cities = Table(
    "cities",
    obj_metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String, nullable=False),
    Column(
        "geometry",
        Geometry(
            srid=4326, spatial_index=False, from_text="ST_GeomFromEWKT", name="geometry"
        ),
    ),
    Column(
        "center",
        Geometry(
            "POINT",
            srid=4326,
            spatial_index=False,
            from_text="ST_GeomFromEWKT",
            name="center",
        ),
    ),
    Column("population", Integer),
    Column("created_at", DateTime),
    Column("updated_at", DateTime),
    Column(
        CityDivisionTypeEnum,
        name="city_division_type",
        nullable=False,
        server_default=text("'ADMIN_UNIT_PARENT'::city_division_type"),
    ),
    Column("local_crs", Integer),
    Column("code", Integer),
    Column("region_id", Integer, ForeignKey("regions.id")),
)

administrative_units = Table(
    "administrative_units",
    obj_metadata,
    Column("id", Integer, primary_key=True),
    Column("parent_id", Integer, ForeignKey("administrative_units.id")),
    Column("city_id", Integer, ForeignKey("cities.id")),
    Column("type_id", Integer, ForeignKey("administrative_unit_types.id")),
    Column("name", String),
    Column(
        "geometry",
        Geometry(
            srid=4326, spatial_index=False, from_text="ST_GeomFromEWKT", name="geometry"
        ),
    ),
    Column(
        "center",
        Geometry(
            "POINT",
            srid=4326,
            spatial_index=False,
            from_text="ST_GeomFromEWKT",
            name="center",
        ),
    ),
    Column("population", Integer),
    Column("created_at", DateTime),
    Column("updated_at", DateTime),
    Column("local_crs", Integer),
    Column("municipality_parent_id", Integer),
)

municipalities = Table(
    "municipalities",
    obj_metadata,
    Column("id", Integer, primary_key=True),
    Column("parent_id", Integer, ForeignKey("municipalities.id")),
    Column("city_id", Integer, ForeignKey("cities.id")),
    Column("type_id", Integer, ForeignKey("municipality_types.id")),
    Column("name", String),
    Column(
        "geometry",
        Geometry(
            srid=4326, spatial_index=False, from_text="ST_GeomFromEWKT", name="geometry"
        ),
    ),
    Column(
        "center",
        Geometry(
            "POINT",
            srid=4326,
            spatial_index=False,
            from_text="ST_GeomFromEWKT",
            name="center",
        ),
    ),
    Column("population", Integer),
    Column("created_at", DateTime),
    Column("updated_at", DateTime),
    Column("admin_unit_parent_id", Integer),
)

blocks = Table(
    "blocks",
    obj_metadata,
    Column("id", Integer, primary_key=True),
    Column("city_id", Integer, ForeignKey("cities.id")),
    Column(
        "geometry",
        Geometry(
            srid=4326, spatial_index=False, from_text="ST_GeomFromEWKT", name="geometry"
        ),
    ),
    Column(
        "center",
        Geometry(
            "POINT",
            srid=4326,
            spatial_index=False,
            from_text="ST_GeomFromEWKT",
            name="center",
        ),
    ),
    Column("population", Integer),
    Column("created_at", DateTime),
    Column("updated_at", DateTime),
    Column("municipality_id", Integer, ForeignKey("municipalities.id")),
    Column("administrative_unit_id", Integer, ForeignKey("administrative_units.id")),
    Column("area", Float),
)

t_all_services = Table(
    "all_services",
    obj_metadata,
    Column("functional_object_id", Integer),
    Column("physical_object_id", Integer),
    Column("building_id", Integer),
    Column(
        "geometry",
        Geometry(
            srid=4326, spatial_index=False, from_text="ST_GeomFromEWKT", name="geometry"
        ),
    ),
    Column(
        "center",
        Geometry(
            "POINT",
            4326,
            spatial_index=False,
            from_text="ST_GeomFromEWKT",
            name="geometry",
        ),
    ),
    Column("city_service_type", String(50)),
    Column("city_service_type_id", Integer),
    Column("city_service_type_code", String(50)),
    Column("city_function", String(50)),
    Column("city_function_id", Integer),
    Column("city_function_code", String(50)),
    Column("infrastructure_type", String(50)),
    Column("infrastructure_type_id", Integer),
    Column("infrastructure_type_code", String(50)),
    Column("service_name", String(200)),
    Column("opening_hours", String(200)),
    Column("website", String(200)),
    Column("phone", String(100)),
    Column("capacity", Integer),
    Column("is_capacity_real", Boolean),
    Column("address", String(200)),
    Column("is_living", Boolean),
    Column("city", String(50)),
    Column("city_id", Integer),
    Column("administrative_unit", String(50)),
    Column("administrative_unit_id", Integer),
    Column("municipality", String(50)),
    Column("municipality_id", Integer),
    Column("block_id", Integer),
    Column("building_properties", JSONB(astext_type=Text())),
    Column("functional_object_properties", JSONB(astext_type=Text())),
    Column("building_modeled", JSONB(astext_type=Text())),
    Column("functional_object_modeled", JSONB(astext_type=Text())),
    Column("functional_object_created_at", DateTime(True)),
    Column("functional_object_updated_at", DateTime(True)),
    Column("physical_object_created_at", DateTime(True)),
    Column("physical_object_updated_at", DateTime(True)),
    Column("updated_at", DateTime(True)),
    Column("created_at", DateTime(True)),
)

t_all_buildings = Table(
    "all_buildings",
    obj_metadata,
    Column("building_id", Integer),
    Column("physical_object_id", Integer),
    Column("address", String(200)),
    Column("project_type", String(100)),
    Column("building_year", SmallInteger),
    Column("repair_years", String(100)),
    Column("building_area", Float),
    Column("living_area", Float),
    Column("storeys_count", SmallInteger),
    Column("central_heating", Boolean),
    Column("central_hotwater", Boolean),
    Column("central_water", Boolean),
    Column("central_electro", Boolean),
    Column("central_gas", Boolean),
    Column("refusechute", Boolean),
    Column("ukname", String(100)),
    Column("lift_count", SmallInteger),
    Column("failure", Boolean),
    Column("is_living", Boolean),
    Column("resident_number", SmallInteger),
    Column("population_balanced", SmallInteger),
    Column("properties", JSONB(astext_type=Text())),
    Column("modeled", JSONB(astext_type=Text())),
    Column("functional_object_id", Integer),
    Column("osm_id", String(50)),
    Column(
        "geometry",
        Geometry(
            srid=4326, spatial_index=False, from_text="ST_GeomFromEWKT", name="geometry"
        ),
    ),
    Column(
        "center",
        Geometry(
            "POINT",
            4326,
            spatial_index=False,
            from_text="ST_GeomFromEWKT",
            name="geometry",
        ),
    ),
    Column("city", String(50)),
    Column("city_id", Integer),
    Column("city_code", String(50)),
    Column("administrative_unit", String(50)),
    Column("administrative_unit_id", Integer),
    Column("municipality", String(50)),
    Column("municipality_id", Integer),
    Column("block_id", Integer),
    Column("functional_object_created_at", DateTime(True)),
    Column("functional_object_updated_at", DateTime(True)),
    Column("physical_object_created_at", DateTime(True)),
    Column("physical_object_updated_at", DateTime(True)),
    Column("updated_at", DateTime(True)),
    Column("created_at", DateTime(True)),
)
