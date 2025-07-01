"""Default routes for basic endpoints."""
import logging
import os
from fastapi import APIRouter
from config.logging import get_app_logger

logger = get_app_logger()
router = APIRouter()


@router.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Hello World"}


@router.get("/check_connection", description="Check connection")
async def check_connection():
    """Check connection to dependencies."""
    try:
        import cv2
        from deepface import DeepFace
        
        # Test with face recognition like original code
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
