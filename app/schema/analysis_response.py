from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

class CoverageResponse(BaseModel):
    coverage_areas: dict
    links_to_project: dict

class EffectItem(BaseModel):
    name: str
    category: Optional[str] = None # TODO: make obligatory str later
    before: float
    after: float
    delta: float
    risk: float

class EffectsResponse(BaseModel):
    effects: List[EffectItem]

class NamedObjectsResponse(BaseModel):
    named_objects: Dict[str, Any]

class ProvisionRiskItem(BaseModel):
    service: str
    risk: float
    provision: float
    category: Optional[str] = None # TODO: make obligatory str later

class ProvisionToRiskResponse(BaseModel):
    provision_to_risk_table: List[ProvisionRiskItem]

class RiskValuesItem(BaseModel):
    category: str
    social_resonance: float = Field(..., alias="Общественный резонанс")
    support_values: float = Field(..., alias="Поддержка ценностей")
    services: str

class RiskValuesResponse(BaseModel):
    values_to_risk_table: List[RiskValuesItem]

class SocialRiskItem(BaseModel):
    services: str
    risk_rating: int
    description: str
    text: Optional[List[str]] = None
    indicators: Optional[List[str]] = None

class SocialRiskResponse(BaseModel):
    social_risk_table: List[SocialRiskItem]

class TextItem(BaseModel):
    category: str
    date: str
    count: int

class TextsResponse(BaseModel):
    texts: List[TextItem]