"""Application configuration module."""
import os
from pydantic import BaseModel


class Settings(BaseModel):
    APP_NAME: str = "Smart Farming Assistant API"
    APP_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    DEBUG: bool = True
    ALLOWED_ORIGINS: list[str] = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:3000",
        "*"
    ]


settings = Settings()
