from pydantic import BaseModel
from typing import Optional


class TerritoryCreate(BaseModel):
    territory_id: int
    name: str
    matched_territory: Optional[str] = None
