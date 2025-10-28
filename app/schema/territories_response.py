from typing import List, Optional
from pydantic import BaseModel

class ProcessStatusStat(BaseModel):
    name: str 
    unprocessed: int
    share: float


class TerritoryStatsResponse(BaseModel):
    territory_id: int
    territory_name: Optional[str]
    messages_total: int
    statuses: List[ProcessStatusStat]


class TerritoriesListResponse(BaseModel):
    territories: List[TerritoryStatsResponse]

class CreateTerritoryResponse(BaseModel):
    territory_id: int
    name: str
    matched_territory: Optional[str] = None
