from pydantic import BaseModel


class Prediction(BaseModel):
    customer_id: str
    label: bool
    percentage: int


class ResponseData(BaseModel):
    data: list[Prediction]


class RequestData(BaseModel):
    start_year: int
    number_q: int
