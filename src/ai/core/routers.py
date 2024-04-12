from fastapi import APIRouter

from ai.api.views import api_router

router = APIRouter()

router.include_router(api_router)
