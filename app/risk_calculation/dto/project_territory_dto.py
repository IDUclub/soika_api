from pydantic import BaseModel, Field

class ProjectTerritoryRequest(BaseModel):
    """
    Class for post request by territory id within the specified project territory.
    """
    territory_id: int = Field(..., examples=[1])
    project_id: int = Field(..., examples=[1])