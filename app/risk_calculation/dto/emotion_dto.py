from pydantic import BaseModel


class EmotionCreate(BaseModel):
    name: str
    emotion_weight: float
