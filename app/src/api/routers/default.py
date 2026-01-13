"""Default routes for basic endpoints."""
import logging
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from ...services.face_service import FaceService
from config.settings import get_settings
from config.logging import get_app_logger

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
        "qdrant_connection": "unknown",
        "minio_connection": "unknown",
        "ready_to_use": False
    }
    
    # 1. Kiểm tra Qdrant (thông qua Database API Port 7005)
    try:
        collections = await face_service.database_client.get_collections()
        if isinstance(collections, list):
            system_status["qdrant_connection"] = "connected"
        else:
            system_status["qdrant_connection"] = "failed"
    except Exception as e:
        logger.error(f"Root check - Qdrant failed: {str(e)}")
        system_status["qdrant_connection"] = f"disconnected ({str(e)})"

    # 2. Kiểm tra MinIO (Storage Port 9000)
    try:
        s3_client = face_service.image_processor._get_s3_client()
        # Thử liệt kê buckets để test kết nối
        s3_client.list_buckets()
        system_status["minio_connection"] = "connected"
    except Exception as e:
        logger.error(f"Root check - MinIO failed: {str(e)}")
        system_status["minio_connection"] = f"disconnected ({str(e)})"

    # 3. Đánh giá tổng thể
    if (system_status["qdrant_connection"] == "connected" and 
        system_status["minio_connection"] == "connected"):
        
        system_status["ready_to_use"] = True
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=system_status
        )
    else:
        # Nếu 1 trong 2 thất bại, trả về lỗi 503 Service Unavailable
        # Điều này báo hiệu cho client biết API chưa dùng được
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