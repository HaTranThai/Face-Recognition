"""
Application configuration settings.
"""
import os
from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")


class Settings(BaseSettings):
    """Application settings."""
    
    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # Environment
    DOCKER_ENV: bool = False
    
    # Database configuration (qdrant_database_FE service)
    QDRANT_DB_HOST: str = "localhost"
    QDRANT_DB_PORT: int = 7005
    
    # Data paths
    CHECKIN_CUSTOMER_PATH: str = "data-face-checkin-customer-images"
    CHECKIN_EMPLOYEE_PATH: str = "data-face-checkin-employee-images"
    REGISTER_CUSTOMER_PATH: str = "data-face-register-customer-images"
    REGISTER_EMPLOYEE_PATH: str = "data-face-register-employee-images"
    
    # Face detection configuration
    KNOWN_FACE_WIDTH: float = 14.3
    EYE_AR_THRESH: float = 0.2
    BLUR_THRESHOLD: int = 100
    LEFT_RIGHT_FACE_THRESHOLD: float = 2.5
    FACE_EXT: int = 10
    CONF_THRESHOLD: float = 0.8
    
    # Eye landmarks
    LEFT_EYE_LANDMARKS: str = "[33, 160, 158, 133, 153, 144]"
    RIGHT_EYE_LANDMARKS: str = "[362, 385, 387, 263, 373, 380]"
    
    # File paths
    MODELS_PATH: str = "models"
    LOGS_PATH: str = "logs"
    SNAPSHOTS_PATH: str = "snapshots"
    STATIC_PATH: str = "static"
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["*"]
    CORS_CREDENTIALS: bool = True
    CORS_METHODS: List[str] = ["*"]
    CORS_HEADERS: List[str] = ["*"]

    API_KEY: str = "mac_dinh"
    
    @property
    def left_eye_landmarks_list(self) -> List[int]:
        """Convert LEFT_EYE_LANDMARKS string to list of integers."""
        return list(map(int, self.LEFT_EYE_LANDMARKS.strip('[]').split(', ')))
    
    @property
    def right_eye_landmarks_list(self) -> List[int]:
        """Convert RIGHT_EYE_LANDMARKS string to list of integers."""
        return list(map(int, self.RIGHT_EYE_LANDMARKS.strip('[]').split(', ')))
    
    class Config:
        env_file = ".env"
        extra = "ignore"  # Ignore extra fields from .env


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
