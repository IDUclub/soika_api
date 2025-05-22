from pydantic import BaseModel


class StatusResponse(BaseModel):
    status: str

class DetailResponse(BaseModel):
    detail: str