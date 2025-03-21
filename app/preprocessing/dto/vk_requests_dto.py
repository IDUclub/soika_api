from datetime import datetime
from pydantic import BaseModel, field_validator, Field
from typing import Optional
from app.common.exceptions.http_exception_wrapper import http_exception

class VKGroupsRequest(BaseModel):
    territory_id: int = Field(
        ...,
        example=1,
        description="ID территории в базе для запроса групп ВК"
    )

class VKTextsRequest(BaseModel):
    territory_id: int = Field(
        ...,
        example=1,
        description="ID территории в базе для запроса текстов"
    )
    to_date: str = Field(
        ...,
        example="2023-01-01",
        description="Дата до которой запрашивать данные (формат YYYY-MM-DD)"
    )
    limit: Optional[int] = Field(
        None,
        example=100,
        description="Лимит количества возвращаемых записей (опционально)"
    )

    @field_validator("to_date")
    @classmethod
    def validate_to_date(cls, value: str) -> str:
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            raise http_exception(status_code=400, msg="to_date должен иметь формат 'YYYY-MM-DD'")
        return value
