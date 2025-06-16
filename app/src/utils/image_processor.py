"""
Image processing utilities for face recognition system.
"""
import asyncio
import base64
import cv2
import datetime
import logging
import numpy as np
import os
import time
from io import BytesIO
from typing import Any

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles image processing and storage operations."""
    
    def __init__(self, config):
        self.config = config
        # Don't initialize aioboto3 session here to avoid version conflicts
        self._s3_client = None
        
    def _get_s3_client(self):
        """Get S3 client for operations.""" 
        # Use minio hostname in Docker, localhost for local development
        docker_env = getattr(self.config, 'DOCKER_ENV', False)
        endpoint_url = 'http://minio:9000' if docker_env else 'http://localhost:9000'
        
        # Use regular boto3 to avoid asyncio issues
        return boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id='minioadmin',
            aws_secret_access_key='minioadmin1245',
            region_name='us-east-1'
        )

    def _upload_to_s3(self, s3_client, bucket_name: str, object_name: str, img_bytes: BytesIO) -> bool:
        """
        Helper method to upload image to S3 synchronously (to be run in thread pool).
        
        Args:
            s3_client: boto3 S3 client
            bucket_name: S3 bucket name
            object_name: S3 object key
            img_bytes: Image data as BytesIO
            
        Returns:
            bool: Success status
        """
        try:
            # Ensure bucket exists
            try:
                s3_client.head_bucket(Bucket=bucket_name)
            except ClientError:
                s3_client.create_bucket(Bucket=bucket_name)
            
            # Reset BytesIO position before upload
            img_bytes.seek(0)
            
            # Upload file
            s3_client.upload_fileobj(
                img_bytes, bucket_name, object_name,
                ExtraArgs={'ContentType': 'image/jpeg'}
            )
            
            logger.info(f"Successfully uploaded image to MinIO: {bucket_name}/{object_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload image to MinIO: {str(e)}")
            return False

    async def save_face_image(self, data, img_decode, face_id: str, name: str, is_checkin: bool = True) -> bool:
        """
        Save face image to MinIO/S3 storage asynchronously.
        
        Args:
            data: Request data containing store_id and role
            img_decode: OpenCV image array
            face_id: Face identifier
            name: Person name
            is_checkin: Whether this is a checkin or registration
            
        Returns:
            bool: Success status
        """
        start_time = time.time()
        img_bytes = None
        img_encoded = None
        
        try:
            # Determine folder based on role and operation type
            if is_checkin:
                folder_save = (self.config.CHECKIN_CUSTOMER_PATH if data.role == '0' 
                             else self.config.CHECKIN_EMPLOYEE_PATH)
            else:
                folder_save = (self.config.REGISTER_CUSTOMER_PATH if data.role == '0'
                             else self.config.REGISTER_EMPLOYEE_PATH)
            
            # Convert OpenCV image to buffer
            encode_start = time.time()
            _, img_encoded = cv2.imencode('.jpg', img_decode)
            img_bytes = BytesIO(img_encoded.tobytes())
            encode_time = time.time() - encode_start
            logger.info(f"[TIMING] {data.store_id} - Image encoding time: {encode_time:.3f}s")
            
            # Generate filename with timestamp
            time_checkin = datetime.datetime.now().strftime("%Y_%m_%d")
            second_checkin = datetime.datetime.now().strftime("%H_%M_%S")
            file_name = f"{face_id}_{name}_{second_checkin}.jpg"
            object_name = f"{data.store_id}/{time_checkin}/{file_name}"
            
            # Upload to MinIO/S3
            upload_start = time.time()
            
            # Run S3 operations in thread pool since we're using sync boto3
            loop = asyncio.get_event_loop()
            s3_client = self._get_s3_client()
            
            # Upload in thread pool
            success = await loop.run_in_executor(
                None, 
                self._upload_to_s3, 
                s3_client, 
                folder_save, 
                object_name, 
                img_bytes
            )
            
            upload_time = time.time() - upload_start
            total_time = time.time() - start_time
            
            logger.info(f"[TIMING] {data.store_id} - Total MinIO upload time: {upload_time:.3f}s")
            logger.info(f"[TIMING] {data.store_id} - Total save image time: {total_time:.3f}s")
            
            return success
            
        except Exception as e:
            logger.error(f"Error in save_face_image: {str(e)}")
            return False
            
        finally:
            # Ensure memory cleanup
            if img_bytes:
                img_bytes.close()
            if img_encoded is not None:
                del img_encoded

    def decode_base64_image(self, img_base64: str) -> np.ndarray:
        """
        Decode base64 image to OpenCV format.
        
        Args:
            img_base64: Base64 encoded image string
            
        Returns:
            np.ndarray: OpenCV image array
        """
        try:
            contents = base64.b64decode(img_base64)
            img_decode = cv2.imdecode(np.frombuffer(contents, np.uint8), -1)
            return img_decode
        except Exception as e:
            logger.error(f"Error decoding base64 image: {str(e)}")
            raise
            
    def resize_image(self, image: np.ndarray, scale_factor: float = 0.5) -> np.ndarray:
        """
        Resize image by scale factor.
        
        Args:
            image: OpenCV image array
            scale_factor: Scale factor for resizing
            
        Returns:
            np.ndarray: Resized image
        """
        return cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
