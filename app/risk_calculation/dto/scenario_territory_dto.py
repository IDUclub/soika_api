from pydantic import BaseModel, Field

class ScenarioTerritoryRequest(BaseModel):
    """
    Class for request by territory id, project id and scenario id.
    """
    territory_id: int = Field(..., examples=[1])
    project_id: int = Field(..., examples=[1])
    scenario_id: int = Field(..., examples=[1])