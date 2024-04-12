from fastapi import APIRouter, status

from .schemas import ResponseData

api_router = APIRouter(prefix='/api/v1')


@api_router.get(
    '/data',
    response_model=ResponseData,
    status_code=status.HTTP_200_OK
)
async def get_data():
    return {
        'data': 'data'
    }
