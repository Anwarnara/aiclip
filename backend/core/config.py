"""
Application Configuration
"""

import os
from functools import lru_cache

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    OUTPUT_DIR: str = os.path.join(BASE_DIR, "output")
    TEMP_DIR: str = os.path.join(BASE_DIR, "temp")
    MODELS_DIR: str = os.path.join(BASE_DIR, "models")
    DOWNLOAD_DIR: str = os.path.join(BASE_DIR, "downloads")
    DATA_DIR: str = os.path.join(BASE_DIR, "data")  # For caching transcription etc

    # API
    CORS_ORIGINS: list = ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"]

    # AI API Configuration
    # AI A (Anthropic format)
    AI_A_URL: str = "http://104.234.26.223:8090/v1/messages"
    AI_A_KEY: str = "test"
    AI_A_MODEL: str = "gemini-3-pro-high"

    # AI B (Gemini raw format)
    AI_B_URL: str = "https://ultyweb.com/account/v1/chat/completions"
    AI_B_KEY: str = "gam_master_u7w3k9x2m5q8r1t4y6p0s3v8n2b5j7h"
    AI_B_MODEL: str = "gemini-2.5-flash"

    # Default AI selection (A or B)
    AI_SELECTED: str = "A"

    # Whisper Configuration
    WHISPER_MODEL: str = "large-v3-turbo"
    WHISPER_DEVICE: str = "cuda"

    # Video Processing
    OUTPUT_WIDTH: int = 1080
    OUTPUT_HEIGHT: int = 1920

    # Face Tracking Defaults
    FACE_DETECTION_CONFIDENCE: float = 0.5
    FACE_TRACKING_SMOOTHING: float = 0.3
    FACE_DETECTION_INTERVAL: int = 5
    FACE_TRACKING_SPEED: float = 0.5

    class Config:
        env_prefix = "AUTOCLIP_"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

# Create directories
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
os.makedirs(settings.TEMP_DIR, exist_ok=True)
os.makedirs(settings.DOWNLOAD_DIR, exist_ok=True)
os.makedirs(settings.DATA_DIR, exist_ok=True)
