"""
Script to register faces from MinIO storage to Qdrant database.
Reads images from data-face-register-employee-images and data-face-register-customer-images buckets.
"""

import os
import base64
import asyncio
import httpx
from pathlib import Path
from typing import List, Dict, Tuple
import re
from datetime import datetime
import boto3
from botocore.exceptions import ClientError

# Configuration
API_BASE_URL = "http://localhost:2024"
MINIO_ENDPOINT = "http://localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin1245"
EMPLOYEE_BUCKET = "data-face-register-employee-images"
CUSTOMER_BUCKET = "data-face-register-customer-images"

# Limits
MAX_CONCURRENT_REQUESTS = 2  # Giảm xuống 2 để tránh quá tải server
REQUEST_TIMEOUT = 60  # Tăng timeout lên 60s vì xử lý face recognition mất thời gian

# Initialize MinIO client
s3_client = boto3.client(
    's3',
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    region_name='us-east-1'
)


def parse_filename(filename: str) -> Tuple[str, str]:
    """
    Parse filename to extract ID and name.
    Format: {id}_{name}_{time}.jpg
    Example: 374_Phuc_15_16_29.jpg -> (374, Phuc)
    """
    try:
        # Remove .jpg extension
        name_without_ext = filename.replace('.jpg', '')
        
        # Split by underscore
        parts = name_without_ext.split('_')
        
        if len(parts) >= 2:
            face_id = parts[0]
            # Name is everything between ID and time (last 3 parts are time HH_MM_SS)
            if len(parts) >= 4:
                name = '_'.join(parts[1:-3])
            else:
                name = parts[1]
            
            return face_id, name
        else:
            return None, None
    except Exception as e:
        print(f"❌ Error parsing filename {filename}: {e}")
        return None, None


def get_image_files(bucket_name: str) -> List[Dict[str, str]]:
    """
    Get all image files from MinIO bucket with their metadata.
    Returns list of dicts with: store_id, date, object_key, id, name
    """
    images = []
    
    try:
        # List all objects in bucket
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name)
        
        for page in pages:
            for obj in page.get('Contents', []):
                object_key = obj['Key']
                
                # Parse object key: {store_id}/{date}/{filename}
                # Example: 325/2025_12_21/374_Phuc_15_16_29.jpg
                parts = object_key.split('/')
                
                if len(parts) == 3:
                    store_id = parts[0]
                    date = parts[1]
                    filename = parts[2]
                    
                    # Only process .jpg files
                    if filename.endswith('.jpg'):
                        face_id, name = parse_filename(filename)
                        
                        if face_id and name:
                            images.append({
                                'store_id': store_id,
                                'date': date,
                                'object_key': object_key,
                                'bucket': bucket_name,
                                'id': face_id,
                                'name': name,
                                'filename': filename
                            })
        
        return images
        
    except ClientError as e:
        print(f"❌ Error listing bucket {bucket_name}: {e}")
        return []
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return []


def image_to_base64(bucket_name: str, object_key: str) -> str:
    """Download image from MinIO and convert to base64 string."""
    try:
        # Download object from MinIO
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        img_data = response['Body'].read()
        
        # Convert to base64
        base64_str = base64.b64encode(img_data).decode('utf-8')
        return base64_str
        
    except ClientError as e:
        print(f"❌ Error downloading from MinIO {bucket_name}/{object_key}: {e}")
        return None
    except Exception as e:
        print(f"❌ Error converting image to base64: {e}")
        return None



async def register_face(client: httpx.AsyncClient, image_data: Dict, role: str, retry_count: int = 0) -> Dict:
    """
    Register a single face via API.
    
    Args:
        client: httpx AsyncClient
        image_data: Dict with store_id, id, name, bucket, object_key
        role: '0' for customer, '1' for employee
    
    Returns:
        Dict with status and message
    """
    try:
        # Convert image to base64
        img_base64 = image_to_base64(image_data['bucket'], image_data['object_key'])
        if not img_base64:
            return {
                'success': False,
                'store_id': image_data['store_id'],
                'id': image_data['id'],
                'name': image_data['name'],
                'error': 'Failed to convert image to base64'
            }
        
        # Prepare request payload
        payload = {
            'store_id': image_data['store_id'],
            'id': image_data['id'],
            'name': image_data['name'],
            'role': role,
            'img_base64': img_base64,
            'is_update': False
        }
        
        # Send request with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await client.post(
                    f"{API_BASE_URL}/create_face_img_base64",
                    json=payload,
                    timeout=REQUEST_TIMEOUT
                )
                
                if response.status_code in [200, 201]:
                    result = response.json()
                    return {
                        'success': True,
                        'store_id': image_data['store_id'],
                        'id': image_data['id'],
                        'name': image_data['name'],
                        'message': result.get('message', 'Success')
                    }
                elif response.status_code == 409:
                    # Face already exists
                    return {
                        'success': False,
                        'store_id': image_data['store_id'],
                        'id': image_data['id'],
                        'name': image_data['name'],
                        'error': 'Face already exists',
                        'skipped': True
                    }
                else:
                    # Other HTTP errors - retry if not last attempt
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    return {
                        'success': False,
                        'store_id': image_data['store_id'],
                        'id': image_data['id'],
                        'name': image_data['name'],
                        'error': f"HTTP {response.status_code}: {response.text}"
                    }
                    
            except httpx.RemoteProtocolError as e:
                # Server disconnected - retry
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return {
                    'success': False,
                    'store_id': image_data['store_id'],
                    'id': image_data['id'],
                    'name': image_data['name'],
                    'error': f"Server disconnected after {max_retries} attempts"
                }
            
    except asyncio.TimeoutError:
        return {
            'success': False,
            'store_id': image_data['store_id'],
            'id': image_data['id'],
            'name': image_data['name'],
            'error': 'Request timeout'
        }
    except Exception as e:
        return {
            'success': False,
            'store_id': image_data['store_id'],
            'id': image_data['id'],
            'name': image_data['name'],
            'error': str(e)
        }


async def process_batch(images: List[Dict], role: str, role_name: str):
    """Process a batch of images with concurrent requests."""
    
    print(f"\n{'='*80}")
    print(f"Processing {role_name.upper()}")
    print(f"{'='*80}")
    print(f"Total images to process: {len(images)}")
    
    if len(images) == 0:
        print("No images found!")
        return
    
    # Statistics
    stats = {
        'total': len(images),
        'success': 0,
        'failed': 0,
        'skipped': 0
    }
    
    # Create async client
    async with httpx.AsyncClient() as client:
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        
        async def process_with_semaphore(img_data):
            async with semaphore:
                # Add small delay between requests to avoid overwhelming server
                await asyncio.sleep(0.1)
                result = await register_face(client, img_data, role)
                
                # Print progress
                if result['success']:
                    stats['success'] += 1
                    print(f"✅ [{stats['success'] + stats['failed'] + stats['skipped']}/{stats['total']}] "
                          f"Store: {result['store_id']}, ID: {result['id']}, Name: {result['name']}")
                elif result.get('skipped'):
                    stats['skipped'] += 1
                    print(f"⏭️  [{stats['success'] + stats['failed'] + stats['skipped']}/{stats['total']}] "
                          f"SKIPPED - Store: {result['store_id']}, ID: {result['id']}, Name: {result['name']} "
                          f"(Already exists)")
                else:
                    stats['failed'] += 1
                    print(f"❌ [{stats['success'] + stats['failed'] + stats['skipped']}/{stats['total']}] "
                          f"FAILED - Store: {result['store_id']}, ID: {result['id']}, Name: {result['name']} "
                          f"- Error: {result.get('error', 'Unknown')}")
                
                return result
        
        # Process all images
        tasks = [process_with_semaphore(img) for img in images]
        results = await asyncio.gather(*tasks)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY - {role_name.upper()}")
    print(f"{'='*80}")
    print(f"Total:    {stats['total']}")
    print(f"✅ Success: {stats['success']}")
    print(f"⏭️  Skipped: {stats['skipped']} (already exists)")
    print(f"❌ Failed:  {stats['failed']}")
    print(f"{'='*80}\n")


async def main():
    """Main function to process all images."""
    
    print(f"\n{'='*80}")
    print(f"FACE REGISTRATION FROM MINIO")
    print(f"{'='*80}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"MinIO endpoint: {MINIO_ENDPOINT}")
    print(f"API base URL: {API_BASE_URL}")
    print(f"Max concurrent requests: {MAX_CONCURRENT_REQUESTS}")
    print(f"{'='*80}\n")
    
    # Get employee images from MinIO
    print(f"📡 Connecting to MinIO and listing employee images...")
    employee_images = get_image_files(EMPLOYEE_BUCKET)
    print(f"📁 Found {len(employee_images)} employee images")
    
    # Get customer images from MinIO
    print(f"📡 Connecting to MinIO and listing customer images...")
    customer_images = get_image_files(CUSTOMER_BUCKET)
    print(f"📁 Found {len(customer_images)} customer images")
    
    # Automatically process both employees and customers
    print(f"\n🚀 Processing both Employees and Customers automatically...")
    print(f"   - Employees: {len(employee_images)} images")
    print(f"   - Customers: {len(customer_images)} images")
    print(f"   - Total: {len(employee_images) + len(customer_images)} images")
    
    # Process employees first
    if len(employee_images) > 0:
        await process_batch(employee_images, '1', 'Employees')
    else:
        print("\n⚠️  No employee images found, skipping...")
    
    # Then process customers
    if len(customer_images) > 0:
        await process_batch(customer_images, '0', 'Customers')
    else:
        print("\n⚠️  No customer images found, skipping...")
    
    print(f"\n{'='*80}")
    print(f"COMPLETED")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    asyncio.run(main())
