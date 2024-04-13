from pydantic import BaseModel


class Prediction(BaseModel):
    id: int
    mark: bool
    percentage: int


class ResponseData(BaseModel):
    data: list[Prediction]


class RequestData(BaseModel):
    start_year: int
    number_q: int
