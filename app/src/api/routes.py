import logging
from fastapi import APIRouter, Depends, HTTPException, status, Security
from fastapi.security import APIKeyHeader
from config.settings import get_settings
from ..services.face_service import FaceService 

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

face_service = FaceService(settings)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key == settings.API_KEY:
        return api_key

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Could not validate credentials - Access Denied"
    )

async def verify_system_health():
    is_qdrant_ok = False
    is_minio_ok = False
    error_details = []

    try:
        col = await face_service.database_client.get_collections()
        if isinstance(col, list):
            is_qdrant_ok = True
        else:
            error_details.append("Qdrant Error")
    except Exception as e:
        error_details.append(f"Qdrant: {str(e)}")

    try:
        face_service.image_processor._get_s3_client().list_buckets()
        is_minio_ok = True
    except Exception as e:
        error_details.append(f"MinIO: {str(e)}")

    if not is_qdrant_ok or not is_minio_ok:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "message": "System is NOT ready. Database or Storage disconnected.",
                "qdrant_status": "OK" if is_qdrant_ok else "DISCONNECTED",
                "minio_status": "OK" if is_minio_ok else "DISCONNECTED",
                "errors": error_details
            }
        )

logger = logging.getLogger(__name__)
router = APIRouter()

router.include_router(default_router)
router.include_router(health_router, prefix="/health")

secure_dependencies = [
    Depends(verify_api_key),    
    Depends(verify_system_health)  
]

router.include_router(face_router, dependencies=secure_dependencies)
router.include_router(database_router, prefix="/qdrant", dependencies=secure_dependencies)
router.include_router(test_router, prefix="/test", dependencies=secure_dependencies)
router.include_router(minio_router, prefix="/minio", dependencies=secure_dependencies)