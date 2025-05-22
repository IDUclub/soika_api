from pydantic import BaseModel, Field

class EmotionCreate(BaseModel):
    name: str = Field(
        ...,
        example="positive",
        description="Название эмоции. Одно из трех: positive, neutral, negative"
    )
    emotion_weight: float = Field(
        ...,
        example=1.0,
        description="Вес эмоции, определяющий её влияние"
    )
