from datetime import datetime
from fastapi import HTTPException, APIRouter
from pydantic import BaseModel, validator
from typing import Optional

class VKGroupsRequest(BaseModel):
    territory_id: int


class VKTextsRequest(BaseModel):
    territory_id: int
    to_date: str
    limit: Optional[int] = None

    @validator("to_date")
    def validate_to_date(cls, value):
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            raise ValueError("to_date должен иметь формат 'YYYY-MM-DD'")
        return value
