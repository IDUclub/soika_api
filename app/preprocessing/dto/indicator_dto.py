from pydantic import BaseModel, Field

class IndicatorCreate(BaseModel):
    name: str = Field(
        ...,
        example="Обеспеченность",
        description="Название показателя"
    )