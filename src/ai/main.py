from contextlib import asynccontextmanager
from fastapi import FastAPI

from core.settings import settings
from core.schemas import HealthModel
from core.routers import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title=settings.APP_TITLE, lifespan=lifespan)

app.include_router(router)


@app.get('api/v1/ai/', response_model=HealthModel, tags=['health'])
async def health_check():
    return {'api': True}
