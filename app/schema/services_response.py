from typing import List, Optional
from pydantic import BaseModel

class ServiceResponse(BaseModel):
    service_id: int
    name: str
    value_id: Optional[int] = None

class ServicesListResponse(BaseModel):
    services: List[ServiceResponse]

class MessageServicePair(BaseModel):
    message_id: int
    service_id: int

class MessageServicePairsResponse(BaseModel):
    message_service_pairs: List[MessageServicePair]

class ExtractServicesResponse(BaseModel):
    detail: str
    processed_messages: int


