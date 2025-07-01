"""Test routes."""
import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from ...services.face_service import FaceService
from config.settings import get_settings
from config.logging import get_app_logger

logger = get_app_logger()
router = APIRouter(tags=["Testing"])

# Initialize settings and service
settings = get_settings()
face_service = FaceService(settings)


@router.get("/basic")
async def test_basic_functionality():
    """Test basic functionality without heavy processing."""
    try:
        # Test database connection
        collections = await face_service.database_client.get_collections()
        
        # Test basic imports
        import cv2
        import numpy as np
        
        # Create a simple test array
        test_array = np.zeros((100, 100, 3), dtype=np.uint8)
        
        return {
            "status": "success",
            "message": "Basic functionality test passed",
            "database_collections": len(collections),
            "opencv_version": cv2.__version__,
            "numpy_available": True,
            "test_array_shape": test_array.shape
        }
    except Exception as e:
        logger.error(f"Basic functionality test failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error", 
                "message": f"Basic test failed: {str(e)}"
            }
        )
