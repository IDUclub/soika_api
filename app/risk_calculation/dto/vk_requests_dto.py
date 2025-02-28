from datetime import datetime
from fastapi import HTTPException, APIRouter
from pydantic import BaseModel, validator

class VKGroupsRequest(BaseModel):
    territory_name: str

    @validator("territory_name")
    def territory_not_empty(cls, value):
        if not value.strip():
            raise ValueError("Параметр territory_name не может быть пустым")
        return value

class VKTextsRequest(BaseModel):
    group_domains: str
    to_date: str

    @validator("group_domains")
    def validate_group_domains(cls, value):
        if not value.strip():
            raise ValueError("group_domains не может быть пустым")
        # Проверяем, что после разделения по запятой каждый домен не пустой
        domains = [d.strip() for d in value.split(",")]
        if not all(domains):
            raise ValueError("Каждый домен в group_domains должен быть непустым")
        return value

    @validator("to_date")
    def validate_to_date(cls, value):
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            raise ValueError("to_date должен иметь формат 'YYYY-MM-DD'")
        return value

