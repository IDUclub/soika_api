from typing import List, Optional
from pydantic import BaseModel

class TerritoryResponse(BaseModel):
    territory_id: int
    name: str
    matched_territory: Optional[str] = None # TODO: maybe need to remove matched territory from attributes

class TerritoriesListResponse(BaseModel):
    territories: List[TerritoryResponse]

class CreateTerritoryResponse(BaseModel):
    territory_id: int
    name: str
    matched_territory: Optional[str] = None
