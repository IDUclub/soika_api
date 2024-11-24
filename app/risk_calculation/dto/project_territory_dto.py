from pydantic import BaseModel, Field
from typing import Optional
from geojson_pydantic import MultiPolygon, Polygon
from app.risk_calculation.dto._polygon_example import poly

class ProjectTerritoryRequest(BaseModel):
    """
    Class for post request by territory id within the specified project territory.
    """
    territory_id: Optional[int] = Field(default=None, examples=[1])
    selection_zone: Optional[Polygon | MultiPolygon] = Field(default=None, examples=[poly])