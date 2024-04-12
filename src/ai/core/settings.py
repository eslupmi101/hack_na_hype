import os
from functools import lru_cache
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from pydantic import BaseSettings

load_dotenv(find_dotenv())

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    APP_TITLE: str = os.getenv('APP_TITLE', 'ai')
    TIME_ZONE: str = os.getenv('TIME_ZONE', 'UTC')


@lru_cache()
def get_settings():
    return Settings()


settings = get_settings()
