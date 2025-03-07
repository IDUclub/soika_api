from pydantic import BaseModel

class IndicatorCreate(BaseModel):
    name: str
