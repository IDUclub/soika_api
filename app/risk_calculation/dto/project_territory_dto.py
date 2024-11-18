from pydantic import BaseModel, Field
from geojson_pydantic import MultiPolygon, Polygon

class ProjectTerritoryRequest(BaseModel):
    """Class for LLMproject territory request with polygon or multipolygon validation."""
    
    selection_zone: Polygon | MultiPolygon = Field(
        ...,
        description="Geometry for the selection zone, must be a valid Polygon or MultiPolygon",
        examples=[
            {
                "type": "Polygon",
                "coordinates": [[[30.0, 10.0], [40.0, 40.0], [20.0, 40.0], [10.0, 20.0], [30.0, 10.0]]]
            },
            {
                "type": "MultiPolygon",
                "coordinates": [
                    [
                        [[30.0, 10.0], [40.0, 40.0], [20.0, 40.0], [10.0, 20.0], [30.0, 10.0]]
                    ],
                    [
                        [[15.0, 5.0], [25.0, 25.0], [5.0, 25.0], [15.0, 5.0]]
                    ]
                ]
            }
        ]
    )
