"""API routes for face recognition system."""
import datetime
import logging
from typing import List
from fastapi import APIRouter, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from ..core.models import CreateFace, FaceRecog, DeleteFace
from ..services.face_service import FaceService
from config.settings import get_settings
import datetime
import os

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize settings and service
settings = get_settings()
face_service = FaceService(settings)


@router.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Hello World"}


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Face Recognition API is running"}


@router.get("/check_connection", description="Check connection")
async def check_connection():
    """Check connection to dependencies."""
    try:
        import cv2
        from deepface import DeepFace
        
        # Test with face recognition like original code
        test_image_path = 'testface.jpg'
        if os.path.exists(test_image_path):
            image = cv2.imread(test_image_path)
            face_is_real = DeepFace.extract_faces(
                img_path=image,
                detector_backend="yolov8",
                align=True,
                anti_spoofing=True,
            )
            logger.info(f"Connection successful, face is real: {face_is_real[0]['is_real']}")
            return face_is_real[0]["is_real"]
        else:
            logger.info("Connection check successful (no test image)")
            return True
    except Exception as e:
        logger.error(f"Connection check failed: {str(e)}")
        return False


@router.post("/face_recog_img_base64", 
            description="Face recognition from image base64; role: 1: Employee, 0: Customer",
            tags=["Face"],
            responses={
                200: {
                    "description": "Successful Response",
                    "content": {
                        "application/json": {
                            "example": {
                                "status": 1,
                                "id": "1",
                                "name": "Nguyen Van A"
                            }
                        }
                    }
                },
                400: {
                    "description": "Bad Request",
                    "content": {
                        "application/json": {
                            "example": {
                                "status": "0 or 2",
                                "message": "message"
                            }
                        }
                    }
                }
            })
async def face_recog_img_base64(data: FaceRecog):
    """
    Recognize a face from base64 image.
    
    - role: 1 for Employee, 0 for Customer
    - img_base64: Base64 encoded image containing face
    - store_id: Store identifier
    """
    try:
        return await face_service.recognize_face(data)
    except Exception as e:
        logger.error(f"Face recognition failed: {str(e)}")
        return JSONResponse(status_code=500, content={
            'status': 2,
            'message': "Internal server error"
        })


@router.post("/create_face_img_base64",
            description="Create face from image base64; id: ID of customer or id of employee; name: Name of customer or id of employee",
            tags=["Face"],
            responses={
                200: {
                    "description": "Successful Response",
                    "content": {
                        "application/json": {
                            "example": {
                                "status": "0, 1 or 2",
                                "message": "message"
                            }
                        }
                    }
                }
            })
async def create_face_img_base64(data: CreateFace):
    """
    Create/register a new face from base64 image.
    
    - id: Person ID
    - name: Person name  
    - role: 1 for Employee, 0 for Customer
    - img_base64: Base64 encoded image containing face
    - store_id: Store identifier
    """
    try:
        logger.info(f"create_face_img_base64 - Received request for {data.name} with id {data.id}")
        result = await face_service.create_face(data)
        logger.info(f"create_face_img_base64 - Request completed successfully")
        return result
    except Exception as e:
        logger.error(f"Face creation failed: {str(e)}", exc_info=True)        
        return JSONResponse(status_code=500, content={
            'status': 2,
            'message': f"Internal server error: {str(e)}"
        })


@router.delete("/delete_employee_face",
            description="Delete face from database; id: ID of customer or id of employee; role: 1: Employee, 0: Customer",
            tags=["Face"],
            responses={
                200: {
                    "description": "Successful Response",
                    "content": {
                        "application/json": {
                            "example": {
                                "status": "0, 1 or 2",
                                "message": "message"
                            }
                        }
                    }
                }
            })
async def delete_employee_face(data: DeleteFace):
    """
    Delete an employee face from the database.
    
    - id: Employee ID to delete
    - store_id: Store identifier
    """
    try:
        return await face_service.delete_face(data)
    except Exception as e:
        logger.error(f"Face deletion failed: {str(e)}")
        return JSONResponse(status_code=500, content={
            'status': 2,
            'message': "Internal server error"
        })


@router.post("/face_recog_img_base64_batch",
            description="Face recognition from image base64 batch; role: 1: Employee, 0: Customer",
            tags=["Face"])
async def face_recog_img_base64_batch(data_list: List[FaceRecog]):
    """
    Batch face recognition from base64 images.
    
    - data_list: List of face recognition requests
    """
    try:
        return await face_service.recognize_face_batch(data_list)
    except Exception as e:
        logger.error(f"Batch face recognition failed: {str(e)}")
        return JSONResponse(status_code=500, content={
            'status': 2,
            'message': "Internal server error"
        })


@router.post("/create_face_img_base64_batch_customers",
            description="Create face from image base64 batch; id: ID of customer; name: Name of customer",
            tags=["Face"])
async def create_face_img_base64_batch_customers(data_list: List[CreateFace]):
    """
    Batch create customer faces from base64 images.
    
    - data_list: List of customer face creation requests
    """
    try:
        return await face_service.create_face_batch_customers(data_list)
    except Exception as e:
        logger.error(f"Batch customer creation failed: {str(e)}")
        return JSONResponse(status_code=500, content={
            'status': 2,
            'message': "Internal server error"
        })


@router.get("/backup_db_one", tags=["Database"])
async def backup_db_one(store_id: str, background_tasks: BackgroundTasks):
    """
    Backup database for a single store.
    
    - store_id: Store identifier to backup
    """
    try:
        return await face_service.backup_db_one(store_id, background_tasks)
    except Exception as e:
        logger.error(f"Database backup failed: {str(e)}")
        return JSONResponse(status_code=500, content={
            'status': 2,
            'message': "Internal server error"
        })


@router.get("/backup_all_db", tags=["Database"])
async def backup_all_db(background_tasks: BackgroundTasks):
    """
    Backup all databases.
    """
    try:
        return await face_service.backup_all_db(background_tasks)
    except Exception as e:
        logger.error(f"All database backup failed: {str(e)}")
        return JSONResponse(status_code=500, content={
            'status': 2,
            'message': "Internal server error"
        })


@router.post("/recover_db", 
            tags=["Database"],
            responses={
                200: {
                    "description": "Successful Response",
                    "content": {
                        "application/json": {
                            "example": {
                                "status": "0, 1 or 2",
                                "message": "message"
                            }
                        }
                    }
                }
            })
async def recover_db(file: UploadFile = File(..., description="File backup")):
    """
    Recover database from backup file.
    
    - file: ZIP backup file to restore from
    """
    try:
        return await face_service.recover_db(file)
    except Exception as e:
        logger.error(f"Database recovery failed: {str(e)}")
        return JSONResponse(status_code=500, content={
            'status': 2,
            'message': "Internal server error"
        })


@router.get("/health/database", tags=["Health"])
async def database_health_check():
    """Check connection to Qdrant database via API."""
    try:
        # Test connection to qdrant_database_FE API
        collections = await face_service.database_client.get_collections()
        
        # Test if we can get collections
        if isinstance(collections, list):
            return {
                "status": "healthy",
                "message": "Database connection successful",
                "database_api": "connected",
                "collections_count": len(collections),
                "collections": collections[:5] if len(collections) > 5 else collections  # Show max 5 collections
            }
        else:
            return {
                "status": "error", 
                "message": "Failed to get collections from database",
                "database_api": "error"
            }
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        return {
            "status": "error",
            "message": f"Database connection failed: {str(e)}",
            "database_api": "disconnected"
        }


@router.get("/health/full", tags=["Health"])
async def full_health_check():
    """Complete system health check including database and MinIO."""
    try:
        # Check basic app health
        app_health = {"status": "healthy", "service": "face_recognition"}
        
        # Check database connection
        db_result = await database_health_check()
        
        # Check MinIO connection
        minio_result = await minio_health_check()
        
        # Check if models directory exists
        models_exist = os.path.exists("models")
        
        # Determine overall status
        overall_healthy = (
            db_result["status"] == "healthy" and 
            minio_result["status"] in ["healthy", "warning"]
        )
        
        return {
            "overall_status": "healthy" if overall_healthy else "degraded",
            "app": app_health,
            "database": db_result,
            "minio": minio_result,
            "models_directory": "exists" if models_exist else "missing",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Full health check failed: {str(e)}")
        return {
            "overall_status": "error",
            "message": f"Health check failed: {str(e)}"
        }


@router.get("/test/basic", tags=["Testing"])
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


@router.get("/health/minio", tags=["Health"])
async def minio_health_check():
    """Check connection to MinIO storage."""
    try:
        # Get image processor instance
        image_processor = face_service.image_processor
        
        # Get S3 client
        s3_client = image_processor._get_s3_client()
        
        # Test basic operations
        test_bucket = "health-check-test"
        
        # Try to list buckets (basic connectivity test)
        try:
            response = s3_client.list_buckets()
            buckets = [bucket['Name'] for bucket in response.get('Buckets', [])]
            bucket_count = len(buckets)
            
            # Test create/delete bucket operation
            try:
                # Try to create a test bucket
                s3_client.create_bucket(Bucket=test_bucket)
                
                # Try to delete the test bucket
                s3_client.delete_bucket(Bucket=test_bucket)
                
                return {
                    "status": "healthy",
                    "message": "MinIO connection successful",
                    "minio_connection": "connected",
                    "buckets_count": bucket_count,
                    "buckets": buckets[:5] if bucket_count > 5 else buckets,  # Show max 5 buckets
                    "operations_test": "passed",
                    "endpoint": s3_client._endpoint.host
                }
                
            except Exception as op_error:
                # Basic connection works but operations might have permission issues
                return {
                    "status": "warning",
                    "message": "MinIO connected but operations limited",
                    "minio_connection": "connected",
                    "buckets_count": bucket_count,
                    "buckets": buckets[:5] if bucket_count > 5 else buckets,
                    "operations_test": "failed",
                    "operations_error": str(op_error),
                    "endpoint": s3_client._endpoint.host
                }
                
        except Exception as conn_error:
            return {
                "status": "error",
                "message": "MinIO connection failed",
                "minio_connection": "disconnected",
                "error": str(conn_error)
            }
            
    except Exception as e:
        logger.error(f"MinIO health check failed: {str(e)}")
        return {
            "status": "error",
            "message": f"MinIO health check failed: {str(e)}",
            "minio_connection": "error"
        }