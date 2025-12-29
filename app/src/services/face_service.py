"""
Face recognition service module containing all business logic for face detection,
recognition, and management operations.
"""

import asyncio
import base64
import datetime
import logging
import os
import shutil
import time
import zipfile
from typing import Optional, Tuple, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import gc
import httpx
from fastapi.responses import JSONResponse, FileResponse
from fastapi import HTTPException

from ..utils.image_processor import ImageProcessor
from ..utils.database_client import DatabaseClient
from ..utils.legacy import (
    distance_face_to_camera, 
    check_face_left_right,
    check_eyes_open,
    check_detect_blur,
    is_full_face,
    adjust_gamma,
    get_embedding,
    detect_face,
    check_condition
)
from ..core.models import CreateFace, FaceRecog, DeleteFace
from config.logging import get_face_logger

logger = get_face_logger()

# Giới hạn kết nối đồng thời 
HTTP_SEMAPHORE = asyncio.Semaphore(10)
PROCESSING_SEMAPHORE = asyncio.Semaphore(10)


class FaceService:
    """Service class for handling face recognition operations."""
    
    def __init__(self, config):
        self.config = config
        self.image_processor = ImageProcessor(config)
        self.database_client = DatabaseClient(config.QDRANT_DB_HOST, config.QDRANT_DB_PORT)
    
    async def detect_and_embed_face(self, data, is_detect_face: bool = True, is_checkin: bool = True) -> Tuple[bool, Any]:
        """
        Detect face in image and generate embedding with comprehensive validation.
        Async version with parallel processing for better performance.
        """
        function_start_time = time.time()

        async with PROCESSING_SEMAPHORE:
            try:
                # Decode image
                decode_start = time.time()
                contents = base64.b64decode(data.img_base64)
                img_decode = cv2.imdecode(np.frombuffer(contents, np.uint8), -1)
                decode_time = time.time() - decode_start
                logger.info(f"[TIMING] {data.store_id} - Image decode time: {decode_time:.3f}s")
                
                logger.info(f"detect_and_embed_face - Image decoded successfully from store {data.store_id}")

                # Perform parallel checks if in checkin mode
                if is_checkin:
                    parallel_checks_start = time.time()
                    loop = asyncio.get_running_loop()
                    with ThreadPoolExecutor() as pool:
                        # Parallel face direction and eyes check
                        face_direction_task = loop.run_in_executor(pool, check_face_left_right, img_decode)
                        eyes_open_task = loop.run_in_executor(pool, check_eyes_open, img_decode)
                        
                        # Wait for face direction check
                        check_flr, message_flr = await face_direction_task
                        parallel_checks_time = time.time() - parallel_checks_start
                        logger.info(f"[TIMING] {data.store_id} - Parallel initial checks time: {parallel_checks_time:.3f}s")
                        logger.info(f"{data.store_id} - Check face left right: {check_flr}")
                        
                        if not check_flr:
                            del contents, img_decode
                            gc.collect()
                            logger.warning(f"{data.store_id} - Face is not aligned properly: {message_flr}")
                            return False, JSONResponse(status_code=400, content={
                                'status': 2,
                                'message': message_flr
                            })
                        
                        # Wait for eyes check
                        check_eyes = await eyes_open_task
                        logger.info(f"{data.store_id} - Check eyes open: {check_eyes}")
                        if not check_eyes:
                            del contents, img_decode
                            gc.collect()
                            logger.warning(f"{data.store_id} - Eyes are closed! Please open your eyes")
                            return False, JSONResponse(status_code=400, content={
                                'status': 2,
                                'message': "Eyes are closed! Please open your eyes"
                            })

                # Face detection
                if is_detect_face:
                    try:
                        face_detection_start = time.time()
                        boxes, scores, distances = detect_face(img_decode)
                        face_detection_time = time.time() - face_detection_start
                        logger.info(f"[TIMING] {data.store_id} - Face detection time: {face_detection_time:.3f}s")
                        logger.info(f"{data.store_id} - Face detected successfully")
                    except Exception as e:
                        del contents, img_decode
                        gc.collect()
                        logger.warning(f"{data.store_id} - Error when detecting face: {str(e)}")
                        if is_checkin:
                            return False, JSONResponse(status_code=500, content={
                                'status': 2,
                                'message': "Error when detecting face! Please try again"
                            })
                        else:
                            return True, (None, img_decode)
                else:
                    scores = [0.9]
                    img_size = img_decode.shape
                    boxes = [[0, 0, img_size[1], img_size[0]]]
                    distances = [50]  # Default distance
                    
            except Exception as e:
                if 'contents' in locals():
                    del contents
                if 'img_decode' in locals():
                    del img_decode
                gc.collect()
                logger.warning(f"{data.store_id} - Error when decoding image: {str(e)}")
                return False, JSONResponse(status_code=500, content={
                    'status': 2,
                    'message': "Error when detecting face! Please try again"
                })
            
            try:
                # Process detected face
                idx_large = np.argmin(distances)
                box = boxes[idx_large]
                x, y, w, h = box
                x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)

                # Distance check for checkin
                if is_checkin:
                    distance_check_start = time.time()
                    distance = distance_face_to_camera((x1, y1, x2, y2), img_decode.shape[1])
                    distance_check_time = time.time() - distance_check_start
                    logger.info(f"[TIMING] {data.store_id} - Distance check time: {distance_check_time:.3f}s")
                    logger.info(f"{data.store_id} - Distance from face to camera: {distance}")
                    
                    if distance < 20:
                        del contents, img_decode
                        gc.collect()
                        logger.warning(f"{data.store_id} - Face is too close! Please move back")
                        return False, JSONResponse(status_code=400, content={
                            'status': 2,
                            'message': "Face is too close! Please move back"
                        })
                    elif distance > 70:
                        del contents, img_decode
                        gc.collect()
                        logger.warning(f"{data.store_id} - Face is too far! Please move forward")
                        return False, JSONResponse(status_code=400, content={
                            'status': 2,
                            'message': "Face is too far! Please move forward"
                        })
                    logger.info(f"{data.store_id} - Face is in the correct distance")
                
                # Crop face
                face_crop_start = time.time()
                face = img_decode[y1:y2, x1:x2]
                face = face.astype('uint8')
                face_crop_time = time.time() - face_crop_start
                logger.info(f"[TIMING] {data.store_id} - Face cropping time: {face_crop_time:.3f}s")
                
                del contents
                
                # Parallel face quality checks for checkin
                if is_checkin:
                    face_quality_checks_start = time.time()
                    loop = asyncio.get_running_loop()
                    with ThreadPoolExecutor() as pool:
                        full_face_task = loop.run_in_executor(pool, is_full_face, face)
                        blur_face_task = loop.run_in_executor(pool, check_detect_blur, face)
                        
                        check_full_face, mess_full_face = await full_face_task
                        check_face_blur = await blur_face_task
                        
                        face_quality_checks_time = time.time() - face_quality_checks_start
                        logger.info(f"[TIMING] {data.store_id} - Face quality checks time: {face_quality_checks_time:.3f}s")
                        
                        logger.info(f"{data.store_id} - Check full face: {check_full_face}")
                        if not check_full_face:
                            del img_decode, face
                            gc.collect()
                            logger.warning(f"{data.store_id} - Face is not full! Please keep your face in the frame")
                            return False, JSONResponse(status_code=400, content={
                                'status': 2,
                                'message': mess_full_face
                            })
                        
                        logger.info(f"{data.store_id} - Check face blur: {check_face_blur}")
                        if not check_face_blur:
                            del img_decode, face
                            gc.collect()
                            logger.warning(f"{data.store_id} - Face is blur! Please keep your face in focus")
                            return False, JSONResponse(status_code=400, content={
                                'status': 2,
                                'message': "Face is blur! Please keep your face in focus"
                            })
                
                # Adjust gamma
                gamma_adjust_start = time.time()
                face = adjust_gamma(face, gamma=1.5)
                gamma_adjust_time = time.time() - gamma_adjust_start
                logger.info(f"[TIMING] {data.store_id} - Gamma adjustment time: {gamma_adjust_time:.3f}s")

                # Generate embedding
                try:
                    embedding_start = time.time()
                    loop = asyncio.get_running_loop()
                    with ThreadPoolExecutor() as pool:
                        emb_task = loop.run_in_executor(pool, lambda: get_embedding(face, img_decode))
                        emb, is_real = await emb_task
                        
                        embedding_time = time.time() - embedding_start
                        logger.info(f"[TIMING] {data.store_id} - Face embedding generation time: {embedding_time:.3f}s")
                        
                        if not is_real and is_checkin:
                            del img_decode, face
                            gc.collect()
                            logger.warning(f"{data.store_id} - Face is not real! Please use your real face")
                            return False, JSONResponse(status_code=400, content={
                                'status': 2,
                                'message': "Face is not real! Please use your real face"
                            })
                except Exception as e:
                    del face, img_decode
                    gc.collect()
                    logger.warning(f"{data.store_id} - Error when encoding face: {str(e)}")
                    return False, JSONResponse(status_code=500, content={
                        'status': 2,
                        'message': "Error! Please try again"
                    })
                
                total_function_time = time.time() - function_start_time
                logger.info(f"[TIMING] {data.store_id} - Total face detection and embedding time: {total_function_time:.3f}s")
                logger.info(f"{data.store_id} - Face is real")
                
                return True, (emb, img_decode)
                
            except Exception as e:
                if 'img_decode' in locals():
                    del img_decode
                if 'face' in locals():
                    del face
                if 'contents' in locals():
                    del contents
                gc.collect()
                logger.warning(f"{data.store_id} - Error in face processing: {str(e)}")
                return False, JSONResponse(status_code=500, content={
                    'status': 2,
                    'message': "Error when processing face! Please try again"
                })
    
    async def search_face(self, collection_name: str, embedding: List[float], store_id: str) -> Dict[str, Any]:
        """Search for a face in the database."""
        start_time = time.time()
        async with HTTP_SEMAPHORE:
            data_search = {
                "collection_name": collection_name,
                "vector_embedding": embedding,
                "store_id": store_id
            }
            
            try:
                search_start = time.time()
                search_results = await self.database_client.search_point(
                    collection_name=collection_name,
                    vector_embedding=embedding,
                    store_id=store_id
                )
                search_time = time.time() - search_start
                total_time = time.time() - start_time
                
                logger.info(f"[TIMING] {store_id} - Face search request time: {search_time:.3f}s")
                logger.info(f"[TIMING] {store_id} - Total face search time: {total_time:.3f}s")
                
                # Convert to expected format
                if search_results:
                    formatted_data = []
                    for result in search_results:
                        formatted_data.append([result['score'], result['payload']])
                    return {"message": "Point found", "data": formatted_data}
                else:
                    return {"message": "Point not found", "data": []}
            except Exception as e:
                logger.error(f"Error in search_face: {str(e)}")
                return {"data": []}
    
    def extract_face_info(self, search_result: Dict[str, Any]) -> Tuple[str, str, str]:
        """Extract face information from search result."""
        try:
            if not search_result or 'data' not in search_result or not search_result['data']:
                return "Unknown", "Unknown", "Unknown"
            
            search_db = search_result['data']
            if not search_db or len(search_db) == 0:
                return "Unknown", "Unknown", "Unknown"
            
            search_item = search_db[0]
            if len(search_item) > 1 and isinstance(search_item[1], dict):
                return (
                    search_item[1].get('id', "Unknown"), 
                    search_item[1].get('name', "Unknown"), 
                    search_item[1].get('time_created', "Unknown")
                )
            return "Unknown", "Unknown", "Unknown"
        except Exception as e:
            logger.error(f"Error extracting face info: {str(e)}")
            return "Unknown", "Unknown", "Unknown"
    
    async def recognize_face(self, data: FaceRecog) -> JSONResponse:
        """Recognize a face from the database."""
        request_start_time = time.time()
        img_decode = None
        
        try:
            logger.info(f"[TIMING] {data.store_id} - Starting face recognition request")
            
            # Check input conditions
            condition_check_start = time.time()
            check_condition_face, message_condition_face = check_condition(data, is_checkin=True)
            condition_check_time = time.time() - condition_check_start
            logger.info(f"[TIMING] {data.store_id} - Condition check time: {condition_check_time:.3f}s")
            
            if not check_condition_face:
                logger.warning(f"recognize_face - {data.store_id} - {message_condition_face}")
                gc.collect()
                return JSONResponse(status_code=400, content={
                    'status': 2,
                    'message': message_condition_face
                })
            
            # Determine collection name and mode
            if data.role == '1':
                collection_name = f'{data.store_id}_Employees'
                is_checkin = True
            elif data.role == '0':
                collection_name = f'{data.store_id}_Customers'
                is_checkin = False
            else:
                gc.collect()
                return JSONResponse(status_code=400, content={
                    'status': 2,
                    'message': "Invalid role"
                })
            
            # Parallel processing: collection check and face detection
            parallel_start = time.time()
            collection_task = self.database_client.ensure_collections_exist(data.store_id)
            detect_task = self.detect_and_embed_face(data, is_checkin=is_checkin)
            
            # Wait for parallel results
            collection_exists, (check_emb, message) = await asyncio.gather(collection_task, detect_task)
            parallel_time = time.time() - parallel_start
            logger.info(f"[TIMING] {data.store_id} - Parallel collection check + face detection time: {parallel_time:.3f}s")
            
            if not collection_exists:
                logger.warning(f"recognize_face - {data.store_id} - Error when create collection")
                gc.collect()
                return JSONResponse(status_code=500, content={
                    'status': 2,
                    'message': "Error when create collection"
                })
            
            if not check_emb:
                logger.warning(f"recognize_face - {data.store_id} - {message}")
                gc.collect()
                return message
            
            emb, img_decode = message
            
            # If no embedding (e.g., no face detected)
            if emb is None:
                save_start = time.time()
                await self.image_processor.save_face_image(data, img_decode, "Unknown", "Unknown")
                save_time = time.time() - save_start
                logger.info(f"[TIMING] {data.store_id} - Save unknown face time: {save_time:.3f}s")
                logger.info(f"recognize_face - {data.store_id} - without embedding")
                
                del img_decode
                gc.collect()
                
                total_time = time.time() - request_start_time
                logger.info(f"[TIMING] {data.store_id} - Total request time (no embedding): {total_time:.3f}s")
                
                return JSONResponse(status_code=200, content={
                    'status': 1,
                    'id': "Unknown",
                    'name': "Unknown"
                })
            
            # Search for face
            search_result = await self.search_face(collection_name, emb, data.store_id)
            
            # Extract face information
            extract_start = time.time()
            face_id, name, time_created = self.extract_face_info(search_result)
            extract_time = time.time() - extract_start
            logger.info(f"[TIMING] {data.store_id} - Extract face info time: {extract_time:.3f}s")
            
            # If face not found
            if face_id == "Unknown" and name == "Unknown":
                del img_decode, emb
                gc.collect()
                
                total_time = time.time() - request_start_time
                logger.info(f"[TIMING] {data.store_id} - Total request time (face not found): {total_time:.3f}s")
                logger.warning(f"recognize_face - {data.store_id} - Face is not existed! Please register your face or checkin again")
                
                return JSONResponse(status_code=404, content={
                    'status': 0,
                    'message': "Face is not existed! Please register your face or checkin again"
                })
            
            # Save recognized face image
            save_start = time.time()
            await self.image_processor.save_face_image(data, img_decode, face_id, name)
            save_time = time.time() - save_start
            logger.info(f"[TIMING] {data.store_id} - Save recognized face time: {save_time:.3f}s")
            
            # Log and return result
            total_time = time.time() - request_start_time
            logger.info(f"[TIMING] {data.store_id} - Total successful request time: {total_time:.3f}s")
            logger.info(f"recognize_face - status: 1, id: {face_id}, name: {name}")
            logger.info(f"recognize_face - {data.store_id} - Face is recognized successfully")
            
            del img_decode, emb
            gc.collect()
            
            return JSONResponse(status_code=200, content={
                'status': 1,
                'id': face_id,
                'name': name
            })
            
        except Exception as e:
            total_time = time.time() - request_start_time
            logger.error(f"[TIMING] {data.store_id} - Total request time (error): {total_time:.3f}s")
            logger.error(f"recognize_face - {data.store_id} - Error: {str(e)}")
            
            # Save image with Unknown info in case of error
            try:
                if img_decode is not None:
                    await self.image_processor.save_face_image(data, img_decode, "Unknown", "Unknown")
            except Exception as save_error:
                logger.error(f"Failed to save image: {str(save_error)}")
                
            if img_decode is not None:
                del img_decode
            gc.collect()
            
            return JSONResponse(status_code=500, content={
                'status': 1,
                'id': "Unknown",
                'name': "Unknown"
            })
    
    async def create_face(self, data: CreateFace, update_face=False) -> JSONResponse:
        """Create a new face entry in the database."""
        if update_face or data.is_update:
            data.is_update = True
            logger_text = "update"
        else:
            logger_text = "create"
    
        create_face_start_time = time.time()
        id_value = data.id
        name_value = data.name 
        store_id = data.store_id
        
        logger.info(f"[TIMING] {store_id} - Starting {logger_text} face request for {name_value} with id {id_value}")
        logger.info(f"{logger_text}_face - {logger_text} face {name_value} with id {id_value} from store {store_id}")
        
        # Check input conditions
        condition_check_start = time.time()
        check_condition_face, message_condition_face = check_condition(data, is_checkin=False)
        condition_check_time = time.time() - condition_check_start
        logger.info(f"[TIMING] {store_id} - Condition check time: {condition_check_time:.3f}s")
        
        if not check_condition_face:
            logger.warning(f"{store_id} - {message_condition_face}")
            gc.collect()
            return JSONResponse(status_code=400, content={
                'status': 2,
                'message': message_condition_face
            })
        
        # Determine collection name
        if data.role == '1':
            collection_name = f'{store_id}_Employees'
        elif data.role == '0':
            collection_name = f'{store_id}_Customers'
        else:
            gc.collect()
            return JSONResponse(status_code=400, content={
                'status': 2,
                'message': "Invalid role"
            })
        
        # Parallel processing: collection check and face detection
        parallel_processing_start = time.time()
        collection_task = self.database_client.ensure_collections_exist(store_id)
        detect_task = self.detect_and_embed_face(data, is_checkin=False)
        
        # Wait for parallel results
        collection_exists, (check_emb, message) = await asyncio.gather(collection_task, detect_task)
        parallel_processing_time = time.time() - parallel_processing_start
        logger.info(f"[TIMING] {store_id} - Parallel collection check + face detection time: {parallel_processing_time:.3f}s")
        
        if not collection_exists:
            logger.warning(f"{logger_text}_face - {store_id} - Error when {logger_text} collection")
            gc.collect()
            return JSONResponse(status_code=500, content={
                'status': 2,
                'message': "Error! Please try again"
            })
        
        if not check_emb:
            logger.warning(f"{logger_text}_face - {store_id} - {message}")
            gc.collect()
            return message
        
        emb, img_decode = message
        
        # If no embedding
        if emb is None:
            save_start = time.time()
            await self.image_processor.save_face_image(data, img_decode, id_value, name_value, is_checkin=False)
            save_time = time.time() - save_start
            logger.info(f"[TIMING] {store_id} - Save face image (no embedding) time: {save_time:.3f}s")
            
            total_time = time.time() - create_face_start_time
            logger.info(f"[TIMING] {store_id} - Total {logger_text} face time (no embedding): {total_time:.3f}s")
            logger.info(f"{logger_text}_face - {store_id} - {logger_text} face {name_value} with id {id_value} successfully without embedding")
            
            del img_decode
            gc.collect()
            
            return JSONResponse(status_code=200, content={
                'status': 1,
                'message': f'{logger_text} face {name_value} with id {id_value} successfully'
            })
        
        # Check if face already exists
        if not data.is_update:
            search_existing_start = time.time()
            search_result = await self.search_face(collection_name, emb, store_id)
            search_existing_time = time.time() - search_existing_start
            logger.info(f"[TIMING] {store_id} - Search existing face time: {search_existing_time:.3f}s")
            
            if search_result.get('data') and len(search_result['data']) > 0:
                total_time = time.time() - create_face_start_time
                logger.info(f"[TIMING] {store_id} - Total {logger_text} face time (face exists): {total_time:.3f}s")
                logger.warning(f"{logger_text}_face - {store_id} - Face is existed! Please use another face")
                return JSONResponse(status_code=409, content={
                    'status': 0,
                    'message': "Face is existed! Please use another face"
                })
        
        # Insert new face
        insert_face_start = time.time()
        
        success = await self.database_client.insert_point(
            collection_name=collection_name,
            vector_embedding=emb,
            id=id_value,
            name=name_value,
            store_id=store_id,
            is_update_id=False
        )
        
        if not success:
            logger.warning(f"{logger_text}_face - {store_id} - Error when insert face")
            return JSONResponse(status_code=500, content={
                'status': 2,
                'message': "Error when insert face"
            })
        
        insert_face_time = time.time() - insert_face_start
        logger.info(f"[TIMING] {store_id} - Insert face to database time: {insert_face_time:.3f}s")
        
        # Save image
        save_image_start = time.time()
        await self.image_processor.save_face_image(data, img_decode, id_value, name_value, is_checkin=False)
        save_image_time = time.time() - save_image_start
        logger.info(f"[TIMING] {store_id} - Save face image time: {save_image_time:.3f}s")
        
        total_time = time.time() - create_face_start_time
        logger.info(f"[TIMING] {store_id} - Total successful {logger_text} face time: {total_time:.3f}s")
        logger.info(f"{logger_text}_face - {store_id} - {logger_text} face {name_value} with id {id_value} successfully")
        
        return JSONResponse(status_code=201, content={
            'status': 1,
            'message': f'{logger_text} face {name_value} with id {id_value} successfully'
        })
    
    async def add_employee_face(self, data: CreateFace, background_tasks) -> JSONResponse:
        background_tasks.add_task(
            self.create_face, data, update_face=True
        )
        return JSONResponse(status_code=201, content={
            'status': 1,
            'message': "Successfully"
        })
    
    async def delete_face(self, data: DeleteFace) -> JSONResponse:
        """
        Delete a face from the database.
        
        Args:
            data: DeleteFace model containing id and store_id
            
        Returns:
            JSONResponse with status and message
        """
        delete_start_time = time.time()
        id_em = data.id
        store_id = data.store_id
        
        logger.info(f"[TIMING] {store_id} - Starting delete face request for id {id_em}")
        
        if id_em is None:
            total_time = time.time() - delete_start_time
            logger.info(f"[TIMING] {store_id} - Total delete face time (missing id): {total_time:.3f}s")
            logger.error(f"delete_employee_face - {store_id} - id is required")
            return JSONResponse(status_code=400, content={
                'status': 2,
                'message': "id is required"
            })

        # Delete from database
        database_delete_start = time.time()
        success = await self.database_client.delete_point(
            collection_name=f"{store_id}_Employees",
            id=id_em
        )
        
        database_delete_time = time.time() - database_delete_start
        logger.info(f"[TIMING] {store_id} - Database delete request time: {database_delete_time:.3f}s")
        
        total_time = time.time() - delete_start_time
        
        if not success:
            logger.info(f"[TIMING] {store_id} - Total delete face time (not found): {total_time:.3f}s")
            logger.error(f"delete_employee_face - {store_id} - Error when delete face")
            return JSONResponse(status_code=404, content={
                'status': 0,
                'message': f"Not found employee with id {id_em}"
            })
        
        logger.info(f"[TIMING] {store_id} - Total successful delete face time: {total_time:.3f}s")
        logger.info(f"delete_employee_face - {store_id} - Delete face with id {id_em} successfully")
        return JSONResponse(status_code=200, content={
            'status': 1,
            'message': f'Delete face with id {id_em} successfully'
        })

    async def recognize_face_batch(self, data_list: List[FaceRecog]) -> JSONResponse:
        """
        Batch face recognition from base64 images.
        
        Args:
            data_list: List of face recognition requests
            
        Returns:
            JSONResponse with batch processing results
        """
        async def process_single_item(data):
            """Process a single face recognition request."""
            img_decode = None
            try:
                # Check condition
                success, message = check_condition(data, is_checkin=True)
                if not success:
                    logger.warning(f"batch - {data.store_id} - {message}")
                    gc.collect()
                    return
                
                # Determine collection name
                if data.role == '1':
                    collection_name = f'{data.store_id}_Employees'
                    is_checkin = True
                elif data.role == '0':
                    collection_name = f'{data.store_id}_Customers'
                    is_checkin = False
                else:
                    logger.warning(f"batch - {data.store_id} - Invalid role")
                    gc.collect()
                    return
                
                # Check collection exists and detect face in parallel
                collection_task = self.database_client.ensure_collections_exist(data.store_id)
                detect_task = self.detect_and_embed_face(data, is_checkin=is_checkin)
                
                # Wait for parallel results
                collection_exists, (check_emb, message) = await asyncio.gather(collection_task, detect_task)
                
                if not collection_exists:
                    logger.warning(f"batch - {data.store_id} - Error with collection")
                    gc.collect()
                    return
                
                if not check_emb:
                    logger.warning(f"batch - {data.store_id} - {message}")
                    gc.collect()
                    return
                
                emb, img_decode, timing_info = message
                
                # If no embedding
                if emb is None:
                    await self.image_processor.save_face_image(data, img_decode, "Unknown", "Unknown")
                    del img_decode
                    gc.collect()
                    return
                
                # Search face
                search_result = await self.search_face(collection_name, emb, data.store_id)
                face_id, name, _ = self.extract_face_info(search_result)
                
                # Save image
                await self.image_processor.save_face_image(data, img_decode, face_id, name)
                
                # Clean up memory
                del img_decode, emb
                gc.collect()
                
            except Exception as e:
                logger.error(f"batch - Error processing item: {str(e)}")
                if 'img_decode' in locals():
                    del img_decode
                if 'emb' in locals():
                    del emb
                gc.collect()
        
        # Process in parallel with semaphore to limit concurrent processing
        async with asyncio.Semaphore(10) as sem:
            async def process_with_sem(data):
                async with sem:
                    return await process_single_item(data)
            
            # Create task list
            tasks = [process_with_sem(data) for data in data_list]
            
            # Process all tasks concurrently with limit
            await asyncio.gather(*tasks)
        
        # Ensure memory cleanup
        gc.collect()
        
        return JSONResponse(status_code=200, content={
            'status': 1,
            'message': "Successfully processed batch"
        })

    async def create_face_batch_customers(self, data_list: List[CreateFace]) -> JSONResponse:
        """
        Batch create customer faces from base64 images.
        
        Args:
            data_list: List of customer face creation requests
            
        Returns:
            JSONResponse with batch processing results
        """
        async def process_single_customer(data):
            """Process a single customer face creation request."""
            try:
                id_value = data.id
                name = data.name
                store_id = data.store_id
                role = data.role
                
                # Skip non-customer users
                if role != '0':
                    return
                
                # Check conditions
                success, message = check_condition(data, is_checkin=False)
                if not success:
                    logger.warning(f"batch_customers - {store_id} - {message}")
                    return
                
                # Decode image and save
                contents = base64.b64decode(data.img_base64)
                img_decode = cv2.imdecode(np.frombuffer(contents, np.uint8), -1)
                
                # Save image in parallel
                save_task = self.image_processor.save_face_image(data, img_decode, id_value, name, is_checkin=False)
                
                # Check collection and detect face in parallel
                collection_name = f'{store_id}_Customers'
                collection_task = self.database_client.ensure_collections_exist(store_id)
                detect_task = self.detect_and_embed_face(data, is_detect_face=True, is_checkin=False)
                
                # Wait for parallel results
                await save_task
                collection_exists, (check_emb, message) = await asyncio.gather(collection_task, detect_task)
                
                if not collection_exists:
                    logger.warning(f"batch_customers - {store_id} - Error with collection")
                    return
                
                if not check_emb:
                    logger.warning(f"batch_customers - {store_id} - {message}")
                    return
                
                emb, img_decode, timing_info = message
                
                # Skip if no embedding
                if emb is None:
                    return
                    
                # Check if face already exists
                search_result = await self.search_face(collection_name, emb, store_id)
                
                if search_result.get('data') and len(search_result['data']) > 0:
                    logger.warning(f"batch_customers - {store_id} - Face already exists for {id_value}")
                    return
                
                # Insert new face into database
                success = await self.database_client.insert_point(
                    collection_name=collection_name,
                    vector_embedding=emb,
                    id=id_value,
                    name=name,
                    store_id=store_id,
                    is_update_id=False
                )
                
                if not success:
                    logger.warning(f"batch_customers - {store_id} - Error inserting face for {id_value}")
                    return
                
                logger.info(f"batch_customers - {store_id} - Successfully created face for {id_value}")
                
            except Exception as e:
                logger.error(f"batch_customers - Error processing: {str(e)}")
        
        # Process in parallel with semaphore to limit concurrent processing
        async with asyncio.Semaphore(8) as sem:
            async def process_with_sem(data):
                async with sem:
                    return await process_single_customer(data)
            
            # Create task list
            tasks = [process_with_sem(data) for data in data_list]
            
            # Process all tasks concurrently with limit
            await asyncio.gather(*tasks)
        
        # Ensure memory cleanup
        gc.collect()
        
        return JSONResponse(status_code=200, content={
            'status': 1,
            'message': "Successfully processed batch customers"
        })

    async def backup_db_one(self, store_id: str, background_tasks) -> JSONResponse:
        """
        Backup database for a single store.
        
        Args:
            store_id: Store identifier to backup
            background_tasks: FastAPI background tasks
            
        Returns:
            FileResponse with backup ZIP file
        """
        
        backup_start_time = time.time()
        logger.info(f"[TIMING] {store_id} - Starting backup database for store {store_id}")
        
        file_path_customer = f'./snapshots/{store_id}_Customers'
        file_path_employee = f'./snapshots/{store_id}_Employees'
        
        # Check snapshot existence
        # check_start = time.time()
        # if not os.path.exists(file_path_customer) or not os.path.exists(file_path_employee):
        #     total_time = time.time() - backup_start_time
        #     logger.info(f"[TIMING] {store_id} - Total backup time (snapshot not found): {total_time:.3f}s")
        #     return JSONResponse(status_code=404, content={
        #         'status': 0,
        #         'message': "Not found snapshot"
        #     })
        # check_time = time.time() - check_start
        # logger.info(f"[TIMING] {store_id} - Snapshot existence check time: {check_time:.3f}s")
        
        # Create snapshots
        try:
            snapshot_create_start = time.time()
            for collection_name in [f'{store_id}_Employees', f'{store_id}_Customers']:
                await self.database_client.create_snapshot(collection_name)
            snapshot_create_time = time.time() - snapshot_create_start
            logger.info(f"[TIMING] {store_id} - Snapshot creation time: {snapshot_create_time:.3f}s")
        except Exception as e:
            pass
        
        time_save = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        zipfile_name = f'snapshots_{store_id}_{time_save}.zip'
        
        try:
            # Create zip file
            zip_create_start = time.time()
            with zipfile.ZipFile(f'./{zipfile_name}', 'w') as zip_file:
                for folder_name in [file_path_customer, file_path_employee]:
                    for root, dirs, files in os.walk(folder_name):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, os.path.join(folder_name, '..'))
                            zip_file.write(file_path, arcname)
            zip_create_time = time.time() - zip_create_start
            logger.info(f"[TIMING] {store_id} - Zip file creation time: {zip_create_time:.3f}s")
            
            background_tasks.add_task(os.remove, f'./{zipfile_name}')
            
            total_time = time.time() - backup_start_time
            logger.info(f"[TIMING] {store_id} - Total successful backup time: {total_time:.3f}s")
            
            return FileResponse(f'./{zipfile_name}', media_type='application/zip', filename=zipfile_name)
        except Exception as e:
            total_time = time.time() - backup_start_time
            logger.info(f"[TIMING] {store_id} - Total backup time (error): {total_time:.3f}s")
            return JSONResponse(status_code=500, content={
                'status': 2,
                'message': str(e)
            })

    async def backup_all_db(self, background_tasks) -> JSONResponse:
        """
        Backup all databases
        
        Args:
            background_tasks: FastAPI background tasks
            
        Returns:
            FileResponse with backup ZIP file containing all databases
        """
        
        backup_all_start_time = time.time()
        logger.info("[TIMING] backup_all - Starting backup all databases")
        
        headers = {
            'Content-Type': 'application/json',
        }
        
        # Get collections list
        get_collections_start = time.time()
        clts = await self.database_client.get_collections()
        logger.info(f"[TIMING] backup_all - Collections list retrieved: {clts}")
        get_collections_time = time.time() - get_collections_start
        logger.info(f"[TIMING] backup_all - Get collections list time: {get_collections_time:.3f}s")
        
        files_path_customer = []
        files_path_employee = []
        
        # Organize collections
        organize_start = time.time()
        for clt in clts:
            if (clt.endswith('Customers')):
                files_path_customer.append(clt)
            elif (clt.endswith('Employees')):
                files_path_employee.append(clt)
        organize_time = time.time() - organize_start
        logger.info(f"[TIMING] backup_all - Collections organization time: {organize_time:.3f}s")
        logger.info(f"[TIMING] backup_all - Customer collections: {files_path_customer}")
        logger.info(f"[TIMING] backup_all - Employee collections: {files_path_employee}")
        
        # Check snapshot existence
        # check_snapshots_start = time.time()
        
        # for file_path_customer, file_path_employee in zip(files_path_customer, files_path_employee):
        #     if not os.path.exists("./snapshots/"+file_path_customer) or not os.path.exists("./snapshots/"+file_path_employee):
        #         total_time = time.time() - backup_all_start_time
        #         logger.info(f"[TIMING] backup_all - Total backup time (snapshot not found): {total_time:.3f}s")
        #         return JSONResponse(status_code=404, content={
        #             'status': 0,
        #             'message': "Not found snapshot"
        #         })
        # check_snapshots_time = time.time() - check_snapshots_start
        # logger.info(f"[TIMING] backup_all - Snapshots existence check time: {check_snapshots_time:.3f}s")
        
        # Create snapshots
        try:
            snapshot_create_start = time.time()
            for clt_name_cus, clt_name_emp in zip(files_path_customer, files_path_employee):
                await self.database_client.create_snapshot(clt_name_cus)
                await self.database_client.create_snapshot(clt_name_emp)
            snapshot_create_time = time.time() - snapshot_create_start
            logger.info(f"[TIMING] backup_all - All snapshots creation time: {snapshot_create_time:.3f}s")
        except Exception as e:
            pass
        
        time_save = datetime.datetime.now().strftime("%Y_%m_%d_")
        zipfile_name = f'snapshots_{time_save}.zip'
        
        try:
            # Create zip file
            zip_create_start = time.time()
            with zipfile.ZipFile(f'./{zipfile_name}', 'w') as zip_file:
                for file_path_customer, file_path_employee in zip(files_path_customer, files_path_employee):
                    for folder_name in ["./snapshots/"+file_path_customer, "./snapshots/"+file_path_employee]:
                        for root, dirs, files in os.walk(folder_name):
                            for file in files:
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, os.path.join(folder_name, '..'))
                                zip_file.write(file_path, arcname)
            zip_create_time = time.time() - zip_create_start
            logger.info(f"[TIMING] backup_all - Zip file creation time: {zip_create_time:.3f}s")
            
            background_tasks.add_task(os.remove, f'./{zipfile_name}')
            
            total_time = time.time() - backup_all_start_time
            logger.info(f"[TIMING] backup_all - Total successful backup all time: {total_time:.3f}s")
            
            return FileResponse(f'./{zipfile_name}', media_type='application/zip', filename=zipfile_name)
        except Exception as e:
            total_time = time.time() - backup_all_start_time
            logger.info(f"[TIMING] backup_all - Total backup all time (error): {total_time:.3f}s")
            return JSONResponse(status_code=500, content={
                'status': 2,
                'message': str(e)
            })

    async def recover_db(self, file) -> JSONResponse:
        """
        Recover database from backup file.
        
        Args:
            file: ZIP backup file to restore from
            
        Returns:
            JSONResponse with recovery status
        """
        
        recover_start_time = time.time()
        logger.info("[TIMING] recover_db - Starting database recovery")
        
        try:
            # Validate file format
            validation_start = time.time()
            if not file.filename.endswith('.zip'):
                total_time = time.time() - recover_start_time
                logger.info(f"[TIMING] recover_db - Total recovery time (invalid format): {total_time:.3f}s")
                raise HTTPException(status_code=400, detail="Invalid file format. Please upload a zip file.")
            validation_time = time.time() - validation_start
            logger.info(f"[TIMING] recover_db - File validation time: {validation_time:.3f}s")

            # Save uploaded file
            file_save_start = time.time()
            temp_zip_path = f"./snapshots/{file.filename}"
            with open(temp_zip_path, "wb") as buffer:
                buffer.write(await file.read())
            file_save_time = time.time() - file_save_start
            logger.info(f"[TIMING] recover_db - File save time: {file_save_time:.3f}s")
            
            # Extract zip file
            extract_start = time.time()
            extract_name = f"extracted_{os.path.splitext(file.filename)[0]}"
            extract_dir = f"snapshots/{extract_name}"
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            os.remove(temp_zip_path)
            extract_time = time.time() - extract_start
            logger.info(f"[TIMING] recover_db - File extraction time: {extract_time:.3f}s")

            # Process extracted files
            process_start = time.time()
            extracted_files = []
            folders = os.listdir(extract_dir)
            for root, dirs, files in os.walk(extract_dir):
                for name in files:
                    extracted_files.append(os.path.join(root, name))
            
            logger.info(f"[TIMING] recover_db - Extracted files: {extracted_files}")
            # Recover snapshots
            recover_snapshots_start = time.time()
            for folder in folders:
                for snapshot_name in os.listdir(os.path.join(extract_dir, folder)):
                    if snapshot_name.endswith('.snapshot'):
                        snapshot_path = os.path.join(extract_name, folder, snapshot_name)
                        logger.info(f"[TIMING] recover_db - Recovering snapshot: {snapshot_path}")
                        success = await self.database_client.recover_snapshot(
                            collection_name=folder,
                            snapshot_name=snapshot_path
                        )
                        
                        if not success:
                            total_time = time.time() - recover_start_time
                            logger.info(f"[TIMING] recover_db - Total recovery time (snapshot error): {total_time:.3f}s")
                            shutil.rmtree(extract_dir)
                            return JSONResponse(status_code=500, content={
                                'status': 2,
                                'message': f"Error recovering snapshot {snapshot_name}"
                            })
            recover_snapshots_time = time.time() - recover_snapshots_start
            logger.info(f"[TIMING] recover_db - Snapshots recovery time: {recover_snapshots_time:.3f}s")
            
            process_time = time.time() - process_start
            logger.info(f"[TIMING] recover_db - File processing time: {process_time:.3f}s")
            
            # Cleanup
            cleanup_start = time.time()
            try:
                shutil.rmtree(extract_dir)
                del extracted_files, folders, folder, snapshot_name, json_post, check
                gc.collect()
                cleanup_time = time.time() - cleanup_start
                logger.info(f"[TIMING] recover_db - Cleanup time: {cleanup_time:.3f}s")
                
                total_time = time.time() - recover_start_time
                logger.info(f"[TIMING] recover_db - Total successful recovery time: {total_time:.3f}s")
                
                return JSONResponse(status_code=200, content={
                    'status': 1,
                    'message': "Recover database successfully"
                })
            except:
                total_time = time.time() - recover_start_time
                logger.info(f"[TIMING] recover_db - Total recovery time (cleanup warning): {total_time:.3f}s")
                return JSONResponse(status_code=200, content={
                    'status': 1,
                    'message': "Recover database successfully"
                })
        except Exception as e:
            total_time = time.time() - recover_start_time
            logger.info(f"[TIMING] recover_db - Total recovery time (error): {total_time:.3f}s")
            return JSONResponse(status_code=500, content={
                'status': 2,
                'message': str(e)
            })
