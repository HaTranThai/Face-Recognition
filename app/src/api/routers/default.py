"""Default routes for basic endpoints."""
import logging
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from ...services.face_service import FaceService
from config.settings import get_settings
from config.logging import get_app_logger
import os

logger = get_app_logger()
router = APIRouter()

settings = get_settings()
face_service = FaceService(settings)

@router.get("/", description="Check connection status of Qdrant and MinIO")
async def root():
    """
    Root endpoint: Checks connections to Qdrant and MinIO.
    Returns 200 OK if all connected, 503 Service Unavailable if any failed.
    """
    system_status = {
        "service": "Face Recognition API (Port 2024)",
        "database_connection": "unknown",
        "storage_connection": "unknown",
        "ready_to_use": False
    }
    
    try:
        collections = await face_service.database_client.get_collections()
        if isinstance(collections, list):
            system_status["database_connection"] = "connected" 
        else:
            system_status["database_connection"] = "failed"
    except Exception as e:
        logger.error(f"Root check - Database (Qdrant) failed: {str(e)}")
        system_status["database_connection"] = "disconnected"

    try:
        s3_client = face_service.image_processor._get_s3_client()
        s3_client.list_buckets()
        system_status["storage_connection"] = "connected"
    except Exception as e:
        logger.error(f"Root check - Storage (MinIO) failed: {str(e)}")
        system_status["storage_connection"] = "disconnected"

    if (system_status.get("database_connection") == "connected" and 
        system_status.get("storage_connection") == "connected"):
        
        system_status["ready_to_use"] = True
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=system_status
        )
    else:
        system_status["ready_to_use"] = False
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=system_status
        )

@router.get("/check_connection", description="Check connection")
async def check_connection():
    """Check connection to dependencies."""
    try:
        import cv2
        from deepface import DeepFace
        
        test_image_path = '/app/static/images/testface.jpg'
        if os.path.exists(test_image_path):
            image = cv2.imread(test_image_path)
            face_is_real = DeepFace.extract_faces(
                img_path=image,
                detector_backend="yolov8",
                align=True,
                anti_spoofing=True,
            )
            logger.info(f"Connection successful, face is real: {face_is_real[0]['is_real']}")
            return {
                'status': 'OK',
                'message': 'Connection successful',
                'face_detected': True,
                'is_real': face_is_real[0]['is_real']
            }
        else:
            logger.warning("Test image not found, returning basic connection status")
            return {
                'status': 'OK',
                'message': 'Basic connection successful (no test image)',
                'face_detected': False
            }
    except Exception as e:
        logger.error(f"Connection check failed: {str(e)}")
        return {
            'status': 'ERROR',
            'message': f'Connection failed: {str(e)}',
            'face_detected': False
        }