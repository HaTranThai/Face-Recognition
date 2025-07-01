"""Database management routes."""
import logging
from fastapi import APIRouter, BackgroundTasks, File, UploadFile
from fastapi.responses import JSONResponse
from ...services.face_service import FaceService
from config.settings import get_settings
from config.logging import get_database_logger

logger = get_database_logger()
router = APIRouter(tags=["Database"])

# Initialize settings and service
settings = get_settings()
face_service = FaceService(settings)


@router.get("/backup_db_one")
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


@router.get("/backup_all_db")
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
