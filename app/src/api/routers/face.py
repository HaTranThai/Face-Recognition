"""Face recognition routes."""
import logging
from typing import List
from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse
from ...core.models import CreateFace, FaceRecog, DeleteFace
from ...services.face_service import FaceService
from config.settings import get_settings
from config.logging import get_face_logger

logger = get_face_logger()
router = APIRouter(tags=["Face"])

# Initialize settings and service
settings = get_settings()
face_service = FaceService(settings)


@router.get("/check_settings")
async def check_settings():
    """
    Check if the settings are correctly configured.
    
    Returns:
        - status: 0 if settings are correct, 1 if not
        - message: Description of the settings status
    """
    return settings

@router.post("/face_recog_img_base64", 
            description="Face recognition from image base64; role: 1: Employee, 0: Customer",
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
            description="Face recognition from image base64 batch; role: 1: Employee, 0: Customer")
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

@router.post("/add_employee_face",
        description="Add employee when they click 'Clock In' or 'Clock Out'; id: ID of employee; name: Name of employee",
        tags=["Face"])
async def add_employee_face(data: CreateFace, background_tasks: BackgroundTasks):
    # Sử dụng phiên bản async cho xử lý nền
    # background_tasks.add_task(process_add_employee_face_async, data)
    try:
        return await face_service.add_employee_face(data, background_tasks)
    except Exception as e:
        logger.error(f"Add employee face failed: {str(e)}")
        return JSONResponse(status_code=500, content={
            'status': 2,
            'message': "Internal server error"
        })

@router.post("/create_face_img_base64_batch_customers",
            description="Create face from image base64 batch; id: ID of customer; name: Name of customer")
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
