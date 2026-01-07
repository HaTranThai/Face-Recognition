import logging
from fastapi import APIRouter, Depends, HTTPException, status, Security
from fastapi.security import APIKeyHeader
from config.settings import get_settings

from .routers import (
    default_router,
    health_router,
    face_router,
    database_router,
    test_router,
    minio_router
)

settings = get_settings()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key == settings.API_KEY:
        return api_key

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Could not validate credentials - Access Denied"
    )

logger = logging.getLogger(__name__)
router = APIRouter()

router.include_router(default_router)
router.include_router(health_router, prefix="/health")

router.include_router(face_router, dependencies=[Depends(verify_api_key)])
router.include_router(database_router, prefix="/qdrant", dependencies=[Depends(verify_api_key)])
router.include_router(test_router, prefix="/test", dependencies=[Depends(verify_api_key)])
router.include_router(minio_router, prefix="/minio", dependencies=[Depends(verify_api_key)])