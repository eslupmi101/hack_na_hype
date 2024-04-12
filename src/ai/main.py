from contextlib import asynccontextmanager
from fastapi import FastAPI

from core.settings import settings
from core.routers import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title=settings.APP_TITLE, lifespan=lifespan)

app.include_router(router)
