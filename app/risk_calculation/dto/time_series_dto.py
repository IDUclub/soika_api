from pydantic import BaseModel, Field
from typing import Literal

class TimeSeriesRequest(BaseModel):
    """
    Class for request by territory id, project id and time period.
    """
    territory_id: int = Field(..., examples=[1])
    project_id: int = Field(..., examples=[1])
    time_period: Literal["day", "week", "month", "year"] = Field(...)