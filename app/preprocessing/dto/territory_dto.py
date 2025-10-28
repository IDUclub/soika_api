from pydantic import BaseModel, Field
from typing import Optional, Literal

class TerritoryGet(BaseModel):
    scope: Literal["all", "with_messages", "with_unprocessed"] = "all"

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