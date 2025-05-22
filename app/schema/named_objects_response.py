from typing import List, Optional
from pydantic import BaseModel

class NamedObjectResponse(BaseModel):
    named_object_id: int
    object_name: str
    object_description: Optional[str] = None
    estimated_location: Optional[str] = None
    accurate_location: Optional[str] = None
    osm_id: int
    count: Optional[int] = None
    osm_tag: Optional[str] = None
    text_id: Optional[int] = None
    geometry: Optional[str] = None  # WKT-представление геометрии
    is_processed: bool

class NamedObjectsListResponse(BaseModel):
    named_objects: List[NamedObjectResponse]

class UploadNamedObjectsResponse(BaseModel):
    inserted_count: int

class ExtractNamedObjectsResponse(BaseModel):
    detail: str
    processed_messages: int
