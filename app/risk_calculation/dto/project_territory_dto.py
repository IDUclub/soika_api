from pydantic import BaseModel, Field


class ProjectTerritoryRequest(BaseModel):
    """
    Class for request by territory id and project id.
    """

    territory_id: int = Field(..., examples=[1])
    project_id: int = Field(..., examples=[1])
