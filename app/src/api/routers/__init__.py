"""API routers package."""

from .default import router as default_router
from .health import router as health_router
from .face import router as face_router
from .database import router as database_router
from .test import router as test_router
from .minio import router as minio_router

__all__ = [
    "default_router",
    "health_router", 
    "face_router",
    "database_router",
    "test_router",
    "minio_router"
]
