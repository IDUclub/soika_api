from typing import List
from pydantic import BaseModel

class IndicatorResponse(BaseModel):
    indicator_id: int
    name: str

class IndicatorsListResponse(BaseModel):
    indicators: List[IndicatorResponse]

class MessageIndicatorPair(BaseModel):
    message_id: int
    indicator_id: int

class MessageIndicatorPairsResponse(BaseModel):
    message_indicator_pairs: List[MessageIndicatorPair]

class ExtractIndicatorsResponse(BaseModel):
    detail: str
    processed_messages: int

class CreateIndicatorResponse(BaseModel):
    indicator_id: int
    name: str
