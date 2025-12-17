from typing import List, Optional
from pydantic import BaseModel, field_validator

class NamedObjectResponse(BaseModel):
    named_object_id: int
    object_name: str
    object_description: Optional[List[str]] = None
    estimated_location: Optional[List[str]] = None
    accurate_location: Optional[str] = None
    osm_id: int
    count: Optional[int] = None
    osm_tag: Optional[str] = None
    text_id: Optional[int] = None
    geometry: Optional[str] = None
    is_processed: bool

    @field_validator("object_description", "estimated_location", mode="before")
    @classmethod
    def _coerce_str_to_list(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            return [v]
        return v

class NamedObjectsListResponse(BaseModel):
    named_objects: List[NamedObjectResponse]

class UploadNamedObjectsResponse(BaseModel):
    inserted_count: int

class ExtractNamedObjectsResponse(BaseModel):
    detail: str
    processed_messages: int
