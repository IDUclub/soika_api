"""create_table

Revision ID: 562662fde66d
Revises: 
Create Date: 2024-09-23 15:29:29.878754

"""
from typing import Sequence as seq, Union

import geoalchemy2
from alembic import op
import sqlalchemy as sa
from fiona.fio.helpers import nullable
from sqlalchemy import Sequence
from sqlalchemy.sql.ddl import CreateSequence

from app.common.db.entities import SoikaTextTypeEnum, SoikaTextSourseTypeEnum


# revision identifiers, used by Alembic.
revision: str = '562662fde66d'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(CreateSequence(Sequence("texts_id_seq")))
    op.create_table(
        "texts",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("text", sa.String(500), nullable=False),
        sa.Column("vkid", sa.Integer, nullable=False),
        sa.Column("public", sa.String(64), nullable=False),
        sa.Column("parent_id", sa.Integer, nullable=False),
        sa.Column("location", sa.String(100), nullable=True, default=None),
        sa.Column("text_type", sa.Enum(SoikaTextSourseTypeEnum), nullable=False),
        sa.Column("text_source", sa.Enum(SoikaTextTypeEnum), nullable=True),
        sa.Column("date", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("city_function", sa.String(64)),
        sa.Column("topic", sa.Integer, nullable=False),
        sa.Column('emotion', sa.String(64), nullable=True),
        sa.Column("service_type", sa.String(64), nullable=True),
        sa.Column("demand_type", sa.String(64), nullable=True),
        sa.Column(
            "center",
            geoalchemy2.types.Geometry(
                spatial_index=False, from_text="ST_GeomFromWKT", name="center", nullable=True
            ),
            nullable=True,
            default=None,
        )
    )


def downgrade() -> None:
    pass
