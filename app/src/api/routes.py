"""API routes for face recognition system."""
import logging
from fastapi import APIRouter
from .routers import (
    default_router,
    health_router,
    face_router,
    database_router,
    test_router,
    minio_router
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Include all routers
router.include_router(default_router)
router.include_router(health_router, prefix="/health")
router.include_router(face_router)
router.include_router(database_router, prefix="/qdrant")
router.include_router(test_router, prefix="/test")
router.include_router(minio_router, prefix="/minio")