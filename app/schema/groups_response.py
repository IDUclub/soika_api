from typing import List
from pydantic import BaseModel

class GroupResponse(BaseModel):
    group_id: int
    name: str
    group_domain: str
    matched_territory: str

class GroupsListResponse(BaseModel):
    groups: List[GroupResponse]


