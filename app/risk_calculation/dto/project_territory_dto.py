from pydantic import BaseModel, Field
from geojson_pydantic import MultiPolygon, Polygon
from app.risk_calculation.dto._polygon_example import poly

class ProjectTerritoryRequest(BaseModel):
    """
    Class for post request by territory id within the specified project territory.
    """
    territory_id: int = Field(..., examples=[1])
    selection_zone: Polygon | MultiPolygon = Field(..., examples=[poly])