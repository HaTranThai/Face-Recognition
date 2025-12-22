"""
Core data models for the face recognition application.
"""
from pydantic import BaseModel
from fastapi import Query
from typing import List, Optional, Dict, Any
from datetime import datetime


# Legacy models for API compatibility
class CreateFace(BaseModel):
    """
    Model for face registration/creation.
    
    Parameters:
    - img_base64: Base64 encoded image containing face
    - id: ID of customer or employee
    - name: Name of customer or employee  
    - role: 1 for Employee, 0 for Customer
    - store_id: Store identifier
    - is_update: Flag to indicate if the face should be updated if it already exists
    """
    img_base64: str = Query(None, description="Ảnh chứa mặt để đăng ký")
    id: str = Query(None, description="ID của khách hàng/ nhân viên")
    name: str = Query(None, description="Tên của khách hàng/ nhân viên")
    role: str = Query(None, description="1: Nhân viên, 0: Khách hàng")
    store_id: str = Query(None, description="ID cửa hàng")
    is_update: bool = Query(False, description="Cập nhật thông tin nếu đã tồn tại")


class DeleteFace(BaseModel):
    """
    Model for face deletion.
    
    Parameters:
    - id: ID of customer or employee to delete
    - store_id: Store identifier
    """
    id: str = Query(None, description="ID của khách hàng")
    store_id: str = Query(None, description="ID cửa hàng")


class FaceRecog(BaseModel):
    """
    Model for face recognition.
    
    Parameters:
    - img_base64: Base64 encoded image containing face
    - role: 1 for Employee, 0 for Customer
    - store_id: Store identifier
    """
    img_base64: str = Query(None, description="Ảnh chứa mặt để nhận diện")
    role: str = Query(None, description="1: Nhân viên, 0: Khách hàng")
    store_id: str = Query(None, description="ID cửa hàng")


# New structured models
class FaceBox(BaseModel):
    """Face bounding box coordinates."""
    x: float
    y: float
    width: float
    height: float
    confidence: float


class FaceFeatures(BaseModel):
    """Face features and quality metrics."""
    is_full_face: bool
    has_mask: bool
    eyes_open: bool
    is_blurry: bool
    face_orientation: str  # "left", "right", "center"
    distance_to_camera: Optional[float] = None
    brightness_level: Optional[float] = None


class DetectedFace(BaseModel):
    """Information about a detected face."""
    bbox: FaceBox
    features: FaceFeatures
    embedding: Optional[List[float]] = None
    face_id: Optional[str] = None


class FaceDetectionResponse(BaseModel):
    """Response for face detection endpoint."""
    success: bool
    message: str
    faces_count: int
    faces: List[DetectedFace]
    processing_time: float
    timestamp: datetime


class PersonMatch(BaseModel):
    """Information about a matched person."""
    person_id: str
    person_name: str
    similarity_score: float
    confidence: float


class FaceRegistrationResponse(BaseModel):
    """Response for face registration endpoint."""
    success: bool
    message: str
    person_id: str
    person_name: str
    collection_name: str
    face_id: str
    processing_time: float
    timestamp: datetime


class FaceComparisonResponse(BaseModel):
    """Response for face comparison endpoint."""
    success: bool
    message: str
    faces_count: int
    matches: List[PersonMatch]
    best_match: Optional[PersonMatch] = None
    processing_time: float
    timestamp: datetime


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = False
    error: str
    error_code: Optional[str] = None
    timestamp: datetime


class CollectionInfo(BaseModel):
    """Information about a face collection."""
    name: str
    person_count: int
    face_count: int
    created_at: datetime
    updated_at: datetime


class QualityCheck(BaseModel):
    """Face quality check results."""
    overall_score: float
    checks: Dict[str, Any]
    passed: bool
    recommendations: List[str]
