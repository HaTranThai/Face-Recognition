"""MinIO backup and restore operations."""
import asyncio
import logging
import os
import tempfile
import zipfile
from datetime import datetime
from io import BytesIO
from typing import List

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from botocore.exceptions import ClientError

from ...services.face_service import FaceService
from config.settings import get_settings
from config.logging import get_minio_logger

logger = get_minio_logger()
router = APIRouter(tags=["MinIO"])

# Initialize settings and service
settings = get_settings()
face_service = FaceService(settings)


@router.get("/list_buckets")
async def list_minio_buckets():
    """
    List all buckets in MinIO storage.
    
    Returns:
        JSONResponse with list of buckets and their metadata
    """
    try:
        s3_client = face_service.image_processor._get_s3_client()
        
        # List all buckets
        response = s3_client.list_buckets()
        buckets = []
        
        for bucket in response.get('Buckets', []):
            bucket_name = bucket['Name']
            creation_date = bucket['CreationDate'].isoformat()
            
            # Get bucket size and object count
            try:
                objects = s3_client.list_objects_v2(Bucket=bucket_name)
                object_count = objects.get('KeyCount', 0)
                total_size = sum(obj.get('Size', 0) for obj in objects.get('Contents', []))
            except ClientError:
                object_count = 0
                total_size = 0
            
            buckets.append({
                "name": bucket_name,
                "creation_date": creation_date,
                "object_count": object_count,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2)
            })
        
        return JSONResponse(content={
            "status": 0,
            "message": "Buckets listed successfully",
            "bucket_count": len(buckets),
            "buckets": buckets
        })
        
    except Exception as e:
        logger.error(f"Failed to list MinIO buckets: {str(e)}")
        return JSONResponse(status_code=500, content={
            "status": 2,
            "message": f"Failed to list buckets: {str(e)}"
        })


@router.get("/backup_bucket/{bucket_name}")
async def backup_minio_bucket(bucket_name: str, background_tasks: BackgroundTasks):
    """
    Backup a specific MinIO bucket to ZIP file.
    
    Args:
        bucket_name: Name of the bucket to backup
        background_tasks: FastAPI background tasks
        
    Returns:
        FileResponse with ZIP backup file
    """
    try:
        s3_client = face_service.image_processor._get_s3_client()
        
        # Check if bucket exists
        try:
            s3_client.head_bucket(Bucket=bucket_name)
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return JSONResponse(status_code=404, content={
                    "status": 1,
                    "message": f"Bucket '{bucket_name}' not found"
                })
            raise
        
        # Create backup directory
        backup_dir = f"./snapshots/minio_backups"
        os.makedirs(backup_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"minio_backup_{bucket_name}_{timestamp}.zip"
        zip_path = os.path.join(backup_dir, zip_filename)
        
        # Create ZIP file with bucket contents
        await _create_bucket_backup(s3_client, bucket_name, zip_path)
        
        # Schedule cleanup after sending file
        background_tasks.add_task(_cleanup_file, zip_path)
                
        return FileResponse(
            path=zip_path,
            filename=zip_filename,
            media_type='application/zip'
        )
        
    except Exception as e:
        logger.error(f"Failed to backup bucket {bucket_name}: {str(e)}")
        return JSONResponse(status_code=500, content={
            "status": 2,
            "message": f"Backup failed: {str(e)}"
        })


@router.get("/backup_all")
async def backup_all_minio(background_tasks: BackgroundTasks):
    """
    Backup all MinIO buckets to a single ZIP file.
    
    Args:
        background_tasks: FastAPI background tasks
        
    Returns:
        FileResponse with ZIP backup file containing all buckets
    """
    try:
        s3_client = face_service.image_processor._get_s3_client()
        
        # Get list of all buckets
        response = s3_client.list_buckets()
        buckets = [bucket['Name'] for bucket in response.get('Buckets', [])]
        
        if not buckets:
            return JSONResponse(content={
                "status": 1,
                "message": "No buckets found to backup"
            })
        
        # Create backup directory
        backup_dir = f"./snapshots/minio_backups"
        os.makedirs(backup_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"minio_backup_all_{timestamp}.zip"
        zip_path = os.path.join(backup_dir, zip_filename)
        
        # Create ZIP file with all buckets
        await _create_all_buckets_backup(s3_client, buckets, zip_path)
        
        # Schedule cleanup after sending file
        background_tasks.add_task(_cleanup_file, zip_path)
        
        # remove snapshots backup_dir
        # os.rmdir(backup_dir)
        
        return FileResponse(
            path=zip_path,
            filename=zip_filename,
            media_type='application/zip'
        )
        
    except Exception as e:
        logger.error(f"Failed to backup all MinIO buckets: {str(e)}")
        return JSONResponse(status_code=500, content={
            "status": 2,
            "message": f"Backup all failed: {str(e)}"
        })


@router.post("/restore_bucket")
async def restore_minio_bucket(
    file: UploadFile = File(..., description="ZIP backup file to restore"),
    overwrite: bool = False
):
    """
    Restore MinIO bucket from backup ZIP file.
    
    Args:
        file: ZIP backup file
        overwrite: Whether to overwrite existing objects
        
    Returns:
        JSONResponse with restore status
    """
    try:
        if not file.filename.endswith('.zip'):
            return JSONResponse(status_code=400, content={
                "status": 1,
                "message": "Only ZIP files are supported"
            })
        
        s3_client = face_service.image_processor._get_s3_client()
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Extract and restore from ZIP file
            restore_result = await _restore_from_backup(s3_client, temp_file_path, overwrite)
            
            return JSONResponse(content={
                "status": 0,
                "message": "Restore completed successfully",
                "details": restore_result
            })
            
        finally:
            # Cleanup temp file
            os.unlink(temp_file_path)
        
    except Exception as e:
        logger.error(f"Failed to restore from backup: {str(e)}")
        return JSONResponse(status_code=500, content={
            "status": 2,
            "message": f"Restore failed: {str(e)}"
        })


@router.post("/sync_buckets")
async def sync_minio_buckets(
    source_bucket: str,
    target_bucket: str,
    delete_extra: bool = False
):
    """
    Sync objects from source bucket to target bucket.
    
    Args:
        source_bucket: Source bucket name
        target_bucket: Target bucket name  
        delete_extra: Whether to delete objects in target that don't exist in source
        
    Returns:
        JSONResponse with sync status
    """
    try:
        s3_client = face_service.image_processor._get_s3_client()
        
        # Check if source bucket exists
        try:
            s3_client.head_bucket(Bucket=source_bucket)
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return JSONResponse(status_code=404, content={
                    "status": 1,
                    "message": f"Source bucket '{source_bucket}' not found"
                })
            raise
        
        # Create target bucket if it doesn't exist
        try:
            s3_client.head_bucket(Bucket=target_bucket)
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                s3_client.create_bucket(Bucket=target_bucket)
                logger.info(f"Created target bucket: {target_bucket}")
        
        # Perform sync
        sync_result = await _sync_buckets(s3_client, source_bucket, target_bucket, delete_extra)
        
        return JSONResponse(content={
            "status": 0,
            "message": "Bucket sync completed successfully",
            "details": sync_result
        })
        
    except Exception as e:
        logger.error(f"Failed to sync buckets: {str(e)}")
        return JSONResponse(status_code=500, content={
            "status": 2,
            "message": f"Sync failed: {str(e)}"
        })


# Helper functions

async def _create_bucket_backup(s3_client, bucket_name: str, zip_path: str):
    """Create backup ZIP file for a single bucket."""
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # List all objects in bucket
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name)
        
        object_count = 0
        for page in pages:
            for obj in page.get('Contents', []):
                key = obj['Key']
                
                # Download object
                response = s3_client.get_object(Bucket=bucket_name, Key=key)
                content = response['Body'].read()
                
                # Add to ZIP with bucket name as folder
                zip_path_in_archive = f"{bucket_name}/{key}"
                zipf.writestr(zip_path_in_archive, content)
                object_count += 1
        
        logger.info(f"Backed up {object_count} objects from bucket {bucket_name}")


async def _create_all_buckets_backup(s3_client, bucket_names: List[str], zip_path: str):
    """Create backup ZIP file for all buckets."""
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        total_objects = 0
        
        for bucket_name in bucket_names:
            # List all objects in bucket
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket_name)
            
            bucket_objects = 0
            for page in pages:
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    
                    # Download object
                    response = s3_client.get_object(Bucket=bucket_name, Key=key)
                    content = response['Body'].read()
                    
                    # Add to ZIP with bucket name as folder
                    zip_path_in_archive = f"{bucket_name}/{key}"
                    zipf.writestr(zip_path_in_archive, content)
                    bucket_objects += 1
                    total_objects += 1
            
            logger.info(f"Backed up {bucket_objects} objects from bucket {bucket_name}")
        
        logger.info(f"Total backed up {total_objects} objects from {len(bucket_names)} buckets")


async def _restore_from_backup(s3_client, zip_path: str, overwrite: bool):
    """Restore buckets from backup ZIP file."""
    restored_objects = 0
    skipped_objects = 0
    created_buckets = []
    
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        for file_info in zipf.filelist:
            if file_info.is_dir():
                continue
            
            # Extract bucket name and object key
            path_parts = file_info.filename.split('/', 1)
            if len(path_parts) != 2:
                continue
            
            bucket_name, object_key = path_parts
            
            # Create bucket if it doesn't exist
            try:
                s3_client.head_bucket(Bucket=bucket_name)
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    s3_client.create_bucket(Bucket=bucket_name)
                    created_buckets.append(bucket_name)
                    logger.info(f"Created bucket: {bucket_name}")
            
            # Check if object exists
            object_exists = False
            if not overwrite:
                try:
                    s3_client.head_object(Bucket=bucket_name, Key=object_key)
                    object_exists = True
                except ClientError:
                    pass
            
            if object_exists and not overwrite:
                skipped_objects += 1
                continue
            
            # Extract and upload object
            content = zipf.read(file_info.filename)
            s3_client.put_object(
                Bucket=bucket_name,
                Key=object_key,
                Body=content
            )
            restored_objects += 1
    
    return {
        "restored_objects": restored_objects,
        "skipped_objects": skipped_objects,
        "created_buckets": created_buckets,
        "overwrite_mode": overwrite
    }


async def _sync_buckets(s3_client, source_bucket: str, target_bucket: str, delete_extra: bool):
    """Sync objects from source to target bucket."""
    copied_objects = 0
    updated_objects = 0
    deleted_objects = 0
    
    # Get source objects
    source_objects = {}
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=source_bucket)
    
    for page in pages:
        for obj in page.get('Contents', []):
            source_objects[obj['Key']] = obj['ETag']
    
    # Get target objects
    target_objects = {}
    try:
        pages = paginator.paginate(Bucket=target_bucket)
        for page in pages:
            for obj in page.get('Contents', []):
                target_objects[obj['Key']] = obj['ETag']
    except ClientError:
        pass
    
    # Copy/update objects from source to target
    for key, etag in source_objects.items():
        if key not in target_objects:
            # Copy new object
            s3_client.copy_object(
                CopySource={'Bucket': source_bucket, 'Key': key},
                Bucket=target_bucket,
                Key=key
            )
            copied_objects += 1
        elif target_objects[key] != etag:
            # Update existing object (ETag different)
            s3_client.copy_object(
                CopySource={'Bucket': source_bucket, 'Key': key},
                Bucket=target_bucket,
                Key=key
            )
            updated_objects += 1
    
    # Delete extra objects in target if requested
    if delete_extra:
        for key in target_objects:
            if key not in source_objects:
                s3_client.delete_object(Bucket=target_bucket, Key=key)
                deleted_objects += 1
    
    return {
        "copied_objects": copied_objects,
        "updated_objects": updated_objects,
        "deleted_objects": deleted_objects,
        "delete_extra_mode": delete_extra
    }


async def _cleanup_file(file_path: str):
    """Remove temporary file after delay."""
    await asyncio.sleep(30)  # Wait 10 seconds before cleanup
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to cleanup file {file_path}: {str(e)}")
