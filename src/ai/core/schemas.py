from pydantic import BaseModel


class HealthModel(BaseModel):
    api: bool
