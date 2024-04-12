from fastapi import APIRouter, status
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import get_swagger_ui_html

from .schemas import ResponseData

api_router = APIRouter(prefix='/api/v1')


@api_router.get(
    '/data',
    response_model=ResponseData,
    status_code=status.HTTP_200_OK,
    summary='Get data',
    description='Retrieve data from the API',
    responses={
        200: {'model': ResponseData, 'description': 'Data retrieved successfully'},
        404: {'description': 'Data not found'}
    }
)
async def get_data():
    return {
        'data': 'data'
    }


# Swagger Docs
def custom_openapi():
    if api_router.openapi_schema:
        return api_router.openapi_schema
    openapi_schema = get_openapi(
        title="Your API",
        version="1.0.0",
        description="Test endpoint",
        routes=api_router.routes,
    )
    api_router.openapi_schema = openapi_schema
    return api_router.openapi_schema


api_router.openapi = custom_openapi


@api_router.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(openapi_url="/openapi.json", title="API documentation")
