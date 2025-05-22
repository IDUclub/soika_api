from pydantic import BaseModel, Field
from typing import Optional


class TerritoryCreate(BaseModel):
    territory_id: int = Field(
        ...,
        example=1,
        description="ID территории"
    )
    name: str = Field(
        ...,
        example="Гатчинский район",
        description="Название территории"
    )
    matched_territory: Optional[str] = Field(
        None,
        example="Гатчинский муниципальный район",
        description="Сопоставленное название территории (если имеется)"
    )