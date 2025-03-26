from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

class MessageResponse(BaseModel):
    message_id: int
    text: str
    date: Optional[datetime]
    views: Optional[int]
    likes: Optional[int]
    reposts: Optional[int]
    type: str
    parent_message_id: int
    group_id: Optional[int]
    emotion_id: Optional[int]
    score: Optional[float]
    geometry: Optional[str]  # WKT-представление геометрии
    location: Optional[str]
    is_processed: bool

class MessagesListResponse(BaseModel):
    messages: List[MessageResponse]

class UploadMessagesResponse(BaseModel):
    inserted_count: int

class VKTextsResponse(BaseModel):
    status: str

class CreateEmotionResponse(BaseModel):
    emotion_id: int
    name: str
    emotion_weight: float
    
class AddressesExtractionResponse(BaseModel):
    status: str
