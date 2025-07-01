"""Health check routes."""
import datetime
import logging
import os
from fastapi import APIRouter
from ...services.face_service import FaceService
from config.settings import get_settings
from config.logging import get_app_logger

logger = get_app_logger()
router = APIRouter(tags=["Health"])

# Initialize settings and service
settings = get_settings()
face_service = FaceService(settings)


@router.get("")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Face Recognition API is running"}


@router.get("/database")
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


@router.get("/minio")
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


@router.get("/full")
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
