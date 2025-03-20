from fastapi import FastAPI, File, UploadFile, Query, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from models.yolo import YOLOv8_face
from pydantic import BaseModel
from deepface import DeepFace
from ultralytics import YOLO as yolofacemask
from typing import List, Union

from dotenv import load_dotenv
from utils import (get_embedding, 
                adjust_gamma, 
                save_face_image, 
                distance_face_to_camera, 
                check_detect_blur,
                check_eyes_open,
                check_face_left_right,
                is_full_face,
                cnc_clt_exist,
                check_condition,
                check_face_mask)

import cv2
import numpy as np
import requests
import zipfile
import os
import base64
import datetime
import shutil
import gc
import boto3
import logging
import httpx
import asyncio
import aioboto3
from io import BytesIO
from functools import partial
from concurrent.futures import ThreadPoolExecutor

load_dotenv(dotenv_path=".env")

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,  # Đặt mức độ log
    format="%(asctime)s - %(levelname)s - %(message)s",  # Định dạng log
    datefmt="%Y-%m-%d %H:%M:%S",  # Định dạng thời gian
    handlers=[
        # logging.StreamHandler(),  # Gửi log ra màn hình
        logging.FileHandler("./logs/face.log", mode="a")  # Gửi log vào file app.log
    ]
)

# Giới hạn kết nối đồng thời 
HTTP_SEMAPHORE = asyncio.Semaphore(20)  # Tối đa 20 kết nối HTTP đồng thời
PROCESSING_SEMAPHORE = asyncio.Semaphore(5)  # Tối đa 5 xử lý hình ảnh đồng thời

# S3 session bất đồng bộ
s3_session = aioboto3.Session()

# Client HTTP bất đồng bộ với cấu hình chung
async def get_http_client():
    return httpx.AsyncClient(timeout=30.0)

# Hàm bất đồng bộ xử lý detect_n_emb_face
async def async_detect_n_emb_face(data, is_detect_face=True, is_checkin=True):
    """
    Phiên bản async của detect_n_emb_face, sử dụng ThreadPoolExecutor 
    cho các tác vụ nặng về CPU (xử lý hình ảnh)
    """
    async with PROCESSING_SEMAPHORE:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(
                pool, 
                partial(detect_n_emb_face, data, is_detect_face, is_checkin)
            )
        return result

# Hàm bất đồng bộ lưu ảnh vào S3/MinIO
async def async_save_face_image(data, img_decode, face_id, name, is_checkin=True):
    """
    Phiên bản async của save_face_image, sử dụng aioboto3
    """
    if is_checkin:
        folder_save = os.getenv("CHECKIN_CUSTOMER_PATH") if data.role == '0' else os.getenv("CHECKIN_EMPLOYEE_PATH")
    else:
        folder_save = os.getenv("REGISTER_CUSTOMER_PATH") if data.role == '0' else os.getenv("REGISTER_EMPLOYEE_PATH")
    
    # Chuyển ảnh OpenCV thành buffer
    _, img_encoded = cv2.imencode('.jpg', img_decode)
    img_bytes = BytesIO(img_encoded.tobytes())
    
    time_checkin = datetime.datetime.now().strftime("%Y_%m_%d")
    second_checkin = datetime.datetime.now().strftime("%H_%M_%S")
    file_name = f"{face_id}_{name}_{second_checkin}.jpg"
    object_name = f"{data.store_id}/{time_checkin}/{file_name}"
    
    async with s3_session.client(
        's3',
        endpoint_url='http://minio:9000',
        aws_access_key_id='minioadmin',
        aws_secret_access_key='minioadmin1245'
    ) as s3:
        try:
            # Đảm bảo bucket tồn tại
            try:
                await s3.head_bucket(Bucket=folder_save)
            except:
                await s3.create_bucket(Bucket=folder_save)
            
            # Upload file
            await s3.upload_fileobj(
                img_bytes, folder_save, object_name,
                ExtraArgs={'ContentType': 'image/jpeg'}
            )
            logger.info(f"Async uploaded image to MinIO: {folder_save} - {object_name}")
            return True
        except Exception as e:
            logger.error(f"Async failed to upload image to MinIO: {str(e)}")
            return False

# Hàm bất đồng bộ kiểm tra và tạo collection
async def async_cnc_clt_exist(store_id):
    async with HTTP_SEMAPHORE:
        async with httpx.AsyncClient() as client:
            try:
                # Kiểm tra collection
                response = await client.get(URL_GET_CLT)
                check_clt = response.json()['collections']
                
                if f"{store_id}_Employees" not in check_clt or f"{store_id}_Customers" not in check_clt:
                    logger.info(f"Creating collections for store {store_id}")
                    # Tạo collections
                    tasks = [
                        client.post(URL_CRE_CLT, json={"collection_name": f"{store_id}_Customers"}),
                        client.post(URL_CRE_CLT, json={"collection_name": f"{store_id}_Employees"})
                    ]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Kiểm tra kết quả
                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            logger.error(f"Failed to create collection: {str(result)}")
                            return False
                        if result.status_code != 201:
                            logger.error(f"Failed to create collection with status {result.status_code}")
                            return False
                return True
                
            except Exception as e:
                logger.error(f"Error in async_cnc_clt_exist: {str(e)}")
                return False

# Hàm bất đồng bộ tìm kiếm khuôn mặt 
async def async_search_face(collection_name, embedding, store_id):
    async with HTTP_SEMAPHORE:
        data_search = {
            "collection_name": collection_name,
            "vector_embedding": embedding,
            "store_id": store_id
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(URL_SEARCH, json=data_search)
                return response.json()
            except Exception as e:
                logger.error(f"Error in async_search_face: {str(e)}")
                return {"data": []}

# Hàm trích xuất thông tin từ kết quả tìm kiếm
def extract_face_info(search_result):
    """Trích xuất thông tin khuôn mặt từ kết quả tìm kiếm."""
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

modelpath ='./models/yolov8n-face.onnx'
confThreshold = 0.8
nmsThreshold = 0.7
YOLOv8_face_detector = YOLOv8_face(modelpath, conf_thres=confThreshold, iou_thres=nmsThreshold)

model_face_mask = yolofacemask("./models/best_face_mask.pt")
logger = logging.getLogger(__name__)


tags_metadata = [
    {
        "name": "Face",
        "description": "APIs for Face"
    },
    {
        "name":"Database",
        "description": "APIs for Database"
    }
]

# app = FastAPI(docs_url=None, redoc_url=None)
app = FastAPI(title="FACE API", description="API for FACE", version="1.0" ,openapi_tags=tags_metadata)

FastDB_HOST = os.getenv("FASTAPI_HOST")
FastDB_PORT = int(os.getenv("FASTAPI_PORT"))

ip_private = f'http://{FastDB_HOST}:{FastDB_PORT}'
URL_SEARCH = os.getenv("URL_SEARCH").format(ip_private = ip_private)
URL_INSERT = os.getenv("URL_INSERT").format(ip_private = ip_private)
URL_DELETE = os.getenv("URL_DELETE").format(ip_private = ip_private)
URL_RECOVER_SNAP = os.getenv("URL_RECOVER_SNAP").format(ip_private = ip_private)
URL_CREATE_SNAP = os.getenv("URL_CREATE_SNAP").format(ip_private = ip_private)
URL_GET_CLT = os.getenv("URL_GET_CLT").format(ip_private = ip_private)
URL_CRE_CLT = os.getenv("URL_CRE_CLT").format(ip_private = ip_private)

FACE_EXT = int(os.getenv("FACE_EXT"))

s3_client = boto3.client(
    's3',
    endpoint_url='http://minio:9000',
    aws_access_key_id='minioadmin',
    aws_secret_access_key='minioadmin1245'
)

# set quyền truy cập cho API
#app.add_middleware(HTTPSRedirectMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CreateFace(BaseModel):
    '''
    Parameters:
    img_base64: str = Query(None, description="Ảnh chứa mặt để đăng ký")
    id: str = Query(None, description="ID của khách hàng/ nhân viên")
    name: str = Query(None, description="Tên của khách hàng/ nhân viên")
    role: str = Query(None, description="1: Nhân viên, 0: Khách hàng")
    store_id: str = Query(None, description="ID cửa hàng")
    '''
    img_base64: str = Query(None, description="Ảnh chứa mặt để đăng ký")
    id:  Union[int, str] = Query(None, description="ID của khách hàng/ nhân viên")
    name: str = Query(None, description="Tên của khách hàng/ nhân viên")
    role: str = Query(None, description="1: Nhân viên, 0: Khách hàng")
    store_id: str = Query(None, description="ID cửa hàng")
    # is_update: str = Query(None, description="1: Update face, 0: Create face")


class DeleteFace(BaseModel):
    id: str = Query(None, description="ID của khách hàng")
    store_id: str = Query(None, description="ID cửa hàng")
    # role: str = Query(None, description="1: Nhân viên, 0: Khách hàng")


class FaceRecog(BaseModel):
    img_base64: str = Query(None, description="Ảnh chứa mặt để nhận diện")
    role: str = Query(None, description="1: Nhân viên, 0: Khách hàng")
    store_id: str = Query(None, description="ID cửa hàng")


def detect_n_emb_face(data, is_detect_face=True, is_checkin=True):
    try:
        contents = data.img_base64
        contents = base64.b64decode(contents)
        img_decode = cv2.imdecode(np.frombuffer(contents, np.uint8), -1)
        
        logger.info(f"detect_n_emb_face - Image decoded successfully from store {data.store_id}")

        if is_checkin == True:
            check_flr, message_flr = check_face_left_right(img_decode)
            # print("check_flr", check_flr)
            logger.info(f"{data.store_id} - Check face left right: {check_flr}")
            if check_flr == False:
                logger.warning(f"{data.store_id} - Face is not aligned properly: {message_flr}", )
                return False,JSONResponse(content={
                    'status': 2,
                    'message': message_flr
                })
        
            check_eyes = check_eyes_open(img_decode)
            # print("check_eyes", check_eyes)
            logger.info(f"{data.store_id} - Check eyes open: {check_eyes}")
            if check_eyes == False:
                logger.warning(f"detect_n_emb_face - {data.store_id} - Eyes are closed! Please open your eyes")
                return False,JSONResponse(content={
                    'status': 2,
                    'message': "Eyes are closed! Please open your eyes"
                })

        if is_detect_face == True:
            try:
                boxes, scores, classIds, kpts = YOLOv8_face_detector.detect(img_decode)
                # print("Scores", scores)
                logger.info(f"{data.store_id} - Face detected successfully")
            except Exception as e:
                logger.warning(f"detect_n_emb_face - {data.store_id} - Error when detecting face: {str(e)}")
                if is_checkin == True:
                    logger.warning(f"detect_n_emb_face - {data.store_id} - Error when detecting face! Please try again")
                    return False,JSONResponse(content={
                        'status': 2,
                        'message': "Error when detecting face! Please try again"
                    })
                else:
                    logger.warning(f"{data.store_id} - Face is not detected")
                    return True, (None, img_decode)
        else:
            scores = [0.9]
            img_size = img_decode.shape
            boxes = [[0, 0, img_size[1], img_size[0]]]
    except:
        # del img_decode, contents
        gc.collect()
        logger.warning(f"{data.store_id} - Error when decoding image")
        return False,JSONResponse(content={
            'status': 2,
            'message': "Error when detecting face! Please try again"
        })
    idx_large = np.argmax(scores)
    box = boxes[idx_large]
    x,y,w,h = box
    x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
    # mở rộng khuôn mặt ra FACE_EXT px 
    if is_detect_face:
        x1 = x1 - FACE_EXT if x1 - FACE_EXT > 0 else 0
        y1 = y1 - FACE_EXT if y1 - FACE_EXT > 0 else 0
        x2 = x2 + FACE_EXT if x2 + FACE_EXT < img_decode.shape[1] else img_decode.shape[1]
        y2 = y2 + FACE_EXT if y2 + FACE_EXT < img_decode.shape[0] else img_decode.shape[0]

    if is_checkin == True:
        distance = distance_face_to_camera((x1, y1, x2, y2), img_decode.shape[1])
        print("distance", distance)
        logger.info(f"{data.store_id} - Distance from face to camera: {distance}")
        if distance < 30 or distance > 70:
            logger.warning(f"{data.store_id} - Face is too close or too far! Please move back or forward")
            return False,JSONResponse(content={
                'status': 2,
                'message': "Face is too close or too far! Please move back or forward"
            })
    
    face = img_decode[y1:y2, x1:x2]
    face = face.astype('uint8')
    
    if is_checkin == True:
        # check_face_is_mask, message_face_is_mask = check_face_mask(model_face_mask, img_decode, box)
        
        # if check_face_is_mask == False:
        #     return False,JSONResponse(content={
        #         'status': 2,
        #         'message': message_face_is_mask
        #     })
        
        check_full_face,mess_full_face = is_full_face(face)
        # print("check_full_face", check_full_face)
        logger.info(f"{data.store_id} - Check full face: {check_full_face}")
        if check_full_face == False:
            logger.warning(f"{data.store_id} - Face is not full! Please keep your face in the frame")
            return False,JSONResponse(content={
                'status': 2,
                'message': mess_full_face
            })
        
        check_face_blur = check_detect_blur(face)
        # print("check_face_blur", check_face_blur)
        logger.info(f"{data.store_id} - Check face blur: {check_face_blur}")
        if check_face_blur == False:
            logger.warning(f"{data.store_id} - Face is blur! Please keep your face in focus")
            return False,JSONResponse(content={
                'status': 2,
                'message': "Face is blur! Please keep your face in focus"
            })
    
    face = adjust_gamma(face, gamma=1.5)

    try:
        emb,is_real = get_embedding(face, img_decode)
        if is_real == False and is_checkin == True:
            logger.warning(f"{data.store_id} - Face is not real! Please use your real face")
            return False,JSONResponse(content={
                'status': 2,
                'message': "Face is not real! Please use your real face"
            })
    except Exception as e:
        del face, img_decode
        gc.collect()
        logger.warning(f"{data.store_id} - Error when encoding face")
        return False,JSONResponse(content={
            'status': 2,
            # 'message': "Error when encoding face"
            "message": "Error! Please try again"
        })
    logger.info(f"{data.store_id} - Face is real")
    return True, (emb, img_decode)


@app.get("/",
        responses={
                200: {
                    "description": "Successful Response",
                    "content": {
                        "application/json": {
                            "example": {
                                "message": "Hello World"
                            }
                        }
                    }
                }
        })
async def root():
    return {"message": "Hello World"}

@app.on_event("startup")
async def startup_event():
    try:
        # Khởi tạo các mô hình và tài nguyên cần thiết
        image = cv2.imread('testface.jpg')
        face_is_real = DeepFace.extract_faces(
            img_path=image,
            detector_backend="yolov8",
            align=True,
            anti_spoofing=True,
        )
        print(face_is_real)
        
        # Tạo thư mục logs nếu chưa có
        os.makedirs("./logs", exist_ok=True)
        
        # Khởi tạo S3 session
        global s3_session
        s3_session = aioboto3.Session()
        
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")

@app.get("/check_connection", description="Check connection")
async def check_connection():
    try:
        image = cv2.imread('testface.jpg')
        face_is_real = DeepFace.extract_faces(
            img_path = image,
            detector_backend = "yolov8",
            align = True,
            anti_spoofing = True,
        )
        print(face_is_real)
        logger.info(f"Connection successful, face is real: {face_is_real[0]['is_real']}")
        return face_is_real[0]["is_real"]
    except Exception as e:
        logger.error(f"Connection failed: {str(e)}")
        return False


@app.post("/face_recog_img_base64",
            description="Face recognition from image base64; role: 1: Employee, 0: Customer", 
            tags=["Face"],
            responses={
                1: {
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
                200: {
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
    try:
        # Kiểm tra điều kiện đầu vào
        check_condition_face, message_condition_face = check_condition(data, is_checkin=True)
        if not check_condition_face:
            logger.warning(f"face_recog_img_base64 - {data.store_id} - {message_condition_face}")
            return JSONResponse(content={
                'status': 2,
                'message': message_condition_face
            })
        
        # Xác định collection name và chế độ
        if data.role == '1':
            collection_name = f'{data.store_id}_Employees'
            is_checkin = True
        elif data.role == '0':
            collection_name = f'{data.store_id}_Customers'
            is_checkin = False
        else:
            return JSONResponse(content={
                'status': 2,
                'message': "Invalid role"
            })
            
        # Kiểm tra và tạo collection nếu cần - async
        collection_exists = await async_cnc_clt_exist(data.store_id)
        if not collection_exists:
            logger.warning(f"face_recog_img_base64 - {data.store_id} - Error when create collection")
            return JSONResponse(content={
                'status': 2,
                'message': "Error when create collection"
            })
        
        # Phát hiện và tính embedding - async wrapper cho CPU-intensive task
        check_emb, message = await async_detect_n_emb_face(data, is_checkin=is_checkin)
        if not check_emb:
            logger.warning(f"face_recog_img_base64 - {data.store_id} - {message}")
            return message
        
        emb, img_decode = message
        
        # Nếu không có embedding (ví dụ: không phát hiện khuôn mặt)
        if emb is None:
            # Lưu ảnh bất đồng bộ
            await async_save_face_image(data, img_decode, "Unknown", "Unknown")
            logger.info(f"face_recog_img_base64 - {data.store_id} - without embedding")
            return JSONResponse(content={
                'status': 1,
                'id': "Unknown",
                'name': "Unknown",
            })
        
        # Tìm kiếm khuôn mặt - bất đồng bộ
        search_result = await async_search_face(collection_name, emb, data.store_id)
        
        # Trích xuất thông tin từ kết quả tìm kiếm
        face_id, name, time_created = extract_face_info(search_result)
        
        # Nếu không tìm thấy khuôn mặt
        if face_id == "Unknown" and name == "Unknown":
            logger.warning(f"face_recog_img_base64 - {data.store_id} - Face is not existed! Please register your face or checkin again")
            return JSONResponse(content={
                'status': 0,
                'message': "Face is not existed! Please register your face or checkin again"
            })
        
        # Lưu ảnh - bất đồng bộ
        await async_save_face_image(data, img_decode, face_id, name)
        
        # Log và trả về kết quả
        logger.info(f"face_recog_img_base64 - status: 1, id: {face_id}, name: {name}")
        logger.info(f"face_recog_img_base64 - {data.store_id} - Face is recognized successfully")
        
        return JSONResponse(content={
            'status': 1,
            'id': face_id,
            'name': name,
        })
        
    except Exception as e:
        logger.error(f"face_recog_img_base64 - {data.store_id} - Error: {str(e)}")
        # Trong trường hợp lỗi, lưu ảnh với thông tin Unknown
        try:
            await async_save_face_image(data, img_decode, "Unknown", "Unknown")
        except Exception as save_error:
            logger.error(f"Failed to save image: {str(save_error)}")
            
        return JSONResponse(content={
            'status': 1,
            'id': "Unknown",
            'name': "Unknown",
        })

@app.post("/face_recog_img_base64_batch",
            description="Face recognition from image base64 batch; role: 1: Employee, 0: Customer", 
            tags=["Face"])
async def face_recog_img_base64_batch(data_list: List[FaceRecog]):
    """
    Xử lý nhận dạng khuôn mặt hàng loạt - bất đồng bộ với asyncio.gather
    """
    async def process_single_item(data):
        try:
            # Kiểm tra điều kiện
            check_condition_face, message_condition_face = check_condition(data, is_checkin=True)
            if not check_condition_face:
                logger.warning(f"batch - {data.store_id} - {message_condition_face}")
                return
            
            # Xác định collection name và chế độ
            if data.role == '1':
                collection_name = f'{data.store_id}_Employees'
                is_checkin = True
            elif data.role == '0':
                collection_name = f'{data.store_id}_Customers'
                is_checkin = False
            else:
                logger.warning(f"batch - {data.store_id} - Invalid role")
                return
            
            # Kiểm tra và tạo collection nếu cần
            collection_exists = await async_cnc_clt_exist(data.store_id)
            if not collection_exists:
                logger.warning(f"batch - {data.store_id} - Error with collection")
                return
            
            # Phát hiện và tính embedding
            check_emb, message = await async_detect_n_emb_face(data, is_checkin=is_checkin)
            if not check_emb:
                logger.warning(f"batch - {data.store_id} - {message}")
                return
            
            emb, img_decode = message
            
            # Nếu không có embedding
            if emb is None:
                await async_save_face_image(data, img_decode, "Unknown", "Unknown")
                return
            
            # Tìm kiếm khuôn mặt
            search_result = await async_search_face(collection_name, emb, data.store_id)
            face_id, name, _ = extract_face_info(search_result)
            
            # Lưu ảnh
            await async_save_face_image(data, img_decode, face_id, name)
            
        except Exception as e:
            logger.error(f"batch - Error processing item: {str(e)}")
    
    # Tạo danh sách các task
    tasks = [process_single_item(data) for data in data_list]
    
    # Xử lý đồng thời tất cả các task
    await asyncio.gather(*tasks)
    
    return JSONResponse(content={
        'status': 1,
        'message': "Successfully processed batch"
    })

@app.post("/create_face_img_base64", 
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
    Tạo khuôn mặt mới từ ảnh base64 (phiên bản async).
    """
    id_value = data.id
    name_value = data.name
    store_id = data.store_id
    
    logger.info(f"create_face_img_base64 - Create face {name_value} with id {id_value} from store {store_id}")
    
    # Kiểm tra điều kiện đầu vào
    check_condition_face, message_condition_face = check_condition(data, is_checkin=True)
    if not check_condition_face:
        logger.warning(f"{store_id} - {message_condition_face}")
        return JSONResponse(content={
            'status': 2,
            'message': message_condition_face
        })
    
    # Xác định collection name và chế độ
    if data.role == '1':
        collection_name = f'{store_id}_Employees'
        is_checkin = False
    elif data.role == '0':
        collection_name = f'{store_id}_Customers'
        is_checkin = False
    else:
        return JSONResponse(content={
            'status': 2,
            'message': "Invalid role"
        })
    
    # Kiểm tra và tạo collection nếu cần - async
    collection_exists = await async_cnc_clt_exist(store_id)
    if not collection_exists:
        logger.warning(f"create_face_img_base64 - {store_id} - Error when create collection")
        return JSONResponse(content={
            'status': 2,
            'message': "Error! Please try again"
        })
    
    # Phát hiện và tính embedding - async wrapper 
    check_emb, message = await async_detect_n_emb_face(data, is_checkin=False)
    if not check_emb:
        logger.warning(f"create_face_img_base64 - {store_id} - {message}")
        return message
    
    emb, img_decode = message
    
    # Nếu không có embedding
    if emb is None:
        await async_save_face_image(data, img_decode, id_value, name_value, is_checkin=False)
        logger.info(f"create_face_img_base64 - {store_id} - Create face {name_value} with id {id_value} successfully without embedding")
        return JSONResponse(content={
            'status': 1,
            'message': f'Create face {name_value} with id {id_value} successfully'
        })
    
    # Kiểm tra khuôn mặt đã tồn tại - async
    search_result = await async_search_face(collection_name, emb, store_id)
    
    if search_result.get('data') and len(search_result['data']) > 0:
        logger.warning(f"create_face_img_base64 - {store_id} - Face is existed! Please use another face")
        return JSONResponse(content={
            'status': 0,
            'message': "Face is existed! Please use another face"
        })
    
    # Thêm khuôn mặt mới - async
    data_insert = {
        "collection_name": collection_name,
        "vector_embedding": emb,
        "id": id_value,
        "name": name_value,
        "store_id": store_id
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(URL_INSERT, json=data_insert)
        
        if response.status_code != 201:
            logger.warning(f"create_face_img_base64 - {store_id} - Error when insert face")
            return JSONResponse(content={
                'status': 2,
                'message': "Error when insert face"
            })
    
    # Lưu ảnh - async
    await async_save_face_image(data, img_decode, id_value, name_value, is_checkin=False)
    
    logger.info(f"create_face_img_base64 - {store_id} - Create face {name_value} with id {id_value} successfully")
    return JSONResponse(content={
        'status': 1,
        'message': f'Create face {name_value} with id {id_value} successfully'
    })

def process_add_employee_face(data: CreateFace):
    try:
        img_base64 = data.img_base64
        id = str(data.id)
        name = data.name
        store_id = data.store_id
        role = data.role
        collection_name=f'{store_id}_Employees'
        
        logger.info(f"process_add_employee_face - {data.store_id} - Add face id {id} - name {name} - role {role}")
        
        check_emb, message = detect_n_emb_face(data, is_checkin=False)# False
        if check_emb == False:
            logger.error(f"process_add_employee_face - {store_id} - {message}")
            return
        emb, img_decode = message
        
        data_insert = {
            "collection_name": collection_name,
            "vector_embedding": emb,
            "id": id,
            "name": name,
            "store_id": store_id
        }
        check = requests.post(URL_INSERT, json=data_insert)
        if check.status_code != 201:
            logger.error(f"process_add_employee_face - {store_id} - {check.content}")
            # print("Error when insert face", check.status_code, check.content)
            # return JSONResponse(content={
            #     'status': 0,
            #     'message': f"Id {id} - error {check.content}"
            # })
            return
        save_face_image(s3_client, data, img_decode, id, name, is_checkin=False)
        logger.info(f"process_add_employee_face - {data.store_id} - Add face {id} successfully")

    except Exception as e:
        logger.error(f"process_add_employee_face - {data.store_id} - {str(e)}")
        return
        # return JSONResponse(content={
        #     'status': 2,
        #     'message': e
        # })

@app.post("/add_employee_face",
        description="Add face from image base64; id: ID of employee; name: Name of employee",
        tags=["Face"])
async def add_employee_face(data: CreateFace, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_add_employee_face, data)
    return JSONResponse(content={
        'status': 1,
        'message': "Successfully"
    })

@app.post("/create_face_img_base64_batch_customers",
            description="Create face from image base64 batch; id: ID of customer; name: Name of customer", 
            tags=["Face"])
async def create_face_img_base64_batch_customers(data_list: List[CreateFace]):
    try:
        for data in data_list:
            id = data.id
            name = data.name
            store_id = data.store_id
            role = data.role
            if role == '1':
                continue
            
            contents = data.img_base64
            contents = base64.b64decode(contents)
            img_decode = cv2.imdecode(np.frombuffer(contents, np.uint8), -1)
            
            save_face_image(s3_client, data, img_decode, id, name, is_checkin=False)
            
            print("1")
            check_condition_face, message_condition_face = check_condition(data, is_checkin=True)
            if check_condition_face == False:
                continue
            
            print("2")
            collection_name=f'{store_id}_Customers'
            is_checkin = False
            
            check_emb, message = detect_n_emb_face(data, is_detect_face= True ,is_checkin=is_checkin) # False
            if check_emb == False:
                continue
            
            emb, img_decode = message
            print("3")
            result_cnc_clt = cnc_clt_exist(data.store_id)
            if result_cnc_clt == False:
                continue
            
            print("4")
            if emb is None:
                save_face_image(s3_client, data, img_decode, id, name, is_checkin=False)
                continue
            # check if id is existed
            data_search = {
                "collection_name": collection_name,
                "vector_embedding": emb,
                "store_id": data.store_id
            }

            print("5")
            search_db = requests.post(URL_SEARCH, json=data_search).json()['data']
            search_db = search_db[0] if len(search_db) > 0 else []

            if len(search_db) > 0:
                continue

            data_insert = {
                    "collection_name": collection_name,
                    "vector_embedding": emb,
                    "id": id,
                    "name": name,
                    "store_id": store_id
                }

            check = requests.post(URL_INSERT, json=data_insert)
            if check.status_code != 201:
                continue
            
            print("Save face")
            save_face_image(s3_client, data, img_decode, id, name, is_checkin=False)

        return JSONResponse(content={
            'status': 1,
            'message': "Successfully"
        })
    except Exception as e:
        return JSONResponse(content={
            'status': 2,
            'message': str(e)
        })

@app.delete("/delete_employee_face", 
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
    Delete a face from the database based on the provided ID and role.

    Parameters:
        data (DeleteFace): The data containing the ID and role of the face to be deleted.

    Returns:
        JSONResponse: The response containing the status and message of the deletion process.
            - status (int): The status code of the response.
            - message (str): The message indicating the result of the deletion process.
    """
    id_em = data.id
    store_id = data.store_id
    if id_em is None:
        logger.error(f"delete_employee_face - {store_id} - id is required")
        return JSONResponse(content={
            'status': 2,
            'message': "id is required"
        })

    data_delete = {
        "collection_name": f"{store_id}_Employees",
        "id": id_em,
    }
    # print(data_delete)
    check = requests.delete(URL_DELETE, json=data_delete)
    if check.status_code != 200:
        logger.error(f"delete_employee_face - {store_id} - Error when delete face")
        return JSONResponse(content={
            'status': 0,
            'message': f"Not found employee with id {id_em}"
        })
    logger.info(f"delete_employee_face - {store_id} - Delete face with id {id_em} successfully")
    return JSONResponse(content={
        'status': 1,
        'message': f'Delete face with id {id_em} successfully'
    })

# update face
# @app.put("/update_face_img_base64", 
#             description="Update face from image base64; id: ID of customer or id of employee; name: Name of customer or id of employee",
#             tags=["Face"],
#             responses={
#                 200: {
#                     "description": "Successful Response",
#                     "content": {
#                         "application/json": {
#                             "example": {
#                                 "status": "0, 1 or 2",
#                                 "message": "message"
#                             }
#                         }
#                     }
#                 }
#             }
#         )

# async def update_face_img_base64(data: CreateFace):
#     """
#     Update a face from an image base64.

#     Parameters:
#         - data (CreateFace): The data containing the image base64, id, name, and role.

#     Returns:
#         - JSONResponse: The response containing the status and message of the face update process.
#             - status (int): The status code of the response.
#             - message (str): The message indicating the result of the face update process.
#     """
#     id = data.id
#     name = data.name
#     role = data.role
#     if id is None or name is None:
#         return JSONResponse(content={
#             'status': 2,
#             'message': "id and name are required"
#         })
        
#     if len(data.img_base64) == 0:
#         return JSONResponse(content={
#             'status': 2,
#             'message': "img_base64 is required"
#         })
    
#     if role == '1':
#         collection_name='Employees'
#     elif role == '0':
#         collection_name='Customers'
#     else:
#         return JSONResponse(content={
#             'status': 2,
#             'message': "role is 0 or 1"
#         })
#     try:
#         contents = data.img_base64
#         contents = base64.b64decode(contents)
#         img_decode = cv2.imdecode(np.frombuffer(contents, np.uint8), -1)
#         img_decode = cv2.resize(img_decode, (0,0), fx=0.5, fy=0.5)
#         boxes, scores, classIds, kpts = YOLOv8_face_detector.detect(img_decode)
#     except Exception as e:
#         del img_decode
#         gc.collect()
#         return JSONResponse(content={
#             'status': 2,
#             'message': "Error when detecting face"
#         })
#     idx_large = np.argmax(scores)
#     box = boxes[idx_large]
#     x,y,w,h = box
#     x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
#     face = img_decode[y1:y2, x1:x2]
#     face = face.astype('uint8')
#     try:
#         emb = get_embedding(face)
#     except Exception as e:
#         del face, img_decode
#         gc.collect()
#         return JSONResponse(content={
#             'status': 2,
#             'message': "Error when encoding face"
#         })
        
#     # check if id is existed
#     data_search = {
#         "collection_name": collection_name,
#         "vector_embedding": emb,
#     }
#     search_db = requests.post(URL_SEARCH, json=data_search).json()['data']
#     search_db = search_db[0] if len(search_db) > 0 else []
#     if len(search_db) == 0:
#         return JSONResponse(content={
#             'status': 0,
#             'message': f'id {id} is not existed'
#         })
#     data_insert = {
#             "collection_name": collection_name,
#             "vector_embedding": emb,
#             "id": id,
#             "name": name,
#             "is_update_id": "true"
#         }
#     check = requests.post(URL_INSERT, json=data_insert)
#     if check.status_code != 201:
#         return JSONResponse(content={
#             'status': 2,
#             'message': "Error when insert face"
#         })
#     del search_db, emb, face, img_decode
#     gc.collect()
#     return JSONResponse(content={
#         'status': 1,
#         'message': f'Update face {name} with id {id} successfully'
#     })
@app.get("/backup_db_one",
            tags=["Database"], 
        )
async def backup_db_one(store_id,background_tasks: BackgroundTasks):
    file_path_customer = f'./snapshots/{store_id}_Customers'
    file_path_employee = f'./snapshots/{store_id}_Employees'
    if not os.path.exists(file_path_customer) or not os.path.exists(file_path_employee):
        return JSONResponse(content={
            'status': 0,
            'message': "Not found snapshot"
        })
    try:
        for collection_name in ['Employees', 'Customers']:
            result = requests.get(URL_CREATE_SNAP+f"/{collection_name}")
    except Exception as e:
        pass
    time_save = datetime.datetime.now().strftime("%Y_%m_%d")
    zipfile_name = f'snapshots_{store_id}_{time_save}.zip'
    try:
        with zipfile.ZipFile(f'./{zipfile_name}', 'w') as zip_file:
            for folder_name in [file_path_customer, file_path_employee]:
                for root, dirs, files in os.walk(folder_name):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, os.path.join(folder_name, '..'))
                        zip_file.write(file_path, arcname)
        background_tasks.add_task(os.remove, f'./{zipfile_name}')
        return FileResponse(f'./{zipfile_name}', media_type='application/zip', filename=zipfile_name)
    except Exception as e:
        return JSONResponse(content={
            'status': 2,
            'message': str(e)
        })

@app.get("/backup_all_db",
            tags=["Database"]
        )
async def backup_all_db(background_tasks: BackgroundTasks):
    headers = {
        'Content-Type': 'application/json',
    }
    clts = requests.get(URL_GET_CLT, headers=headers).json()['collections']
    
    files_path_customer=[]
    files_path_employee=[]
    
    for clt in clts:
        if (clt.endswith('Customers')):
            files_path_customer.append(clt)
        elif (clt.endswith('Employees')):
            files_path_employee.append(clt)
    for file_path_customer, file_path_employee in zip(files_path_customer, files_path_employee):
        if not os.path.exists("./snapshots/"+file_path_customer) or not os.path.exists("./snapshots/"+file_path_employee):
            return JSONResponse(content={
                'status': 0,
                'message': "Not found snapshot"
            })
    
    try:
        for clt_name_cus, clt_name_emp in zip(files_path_customer, files_path_employee):
            result_cus = requests.get(URL_CREATE_SNAP+f"/{clt_name_cus}")
            result_emp = requests.get(URL_CREATE_SNAP+f"/{clt_name_emp}")
    except Exception as e:
        pass
    time_save = datetime.datetime.now().strftime("%Y_%m_%d")
    zipfile_name = f'snapshots_{time_save}.zip'
    try:
        with zipfile.ZipFile(f'./{zipfile_name}', 'w') as zip_file:
            for file_path_customer, file_path_employee in zip(files_path_customer, files_path_employee):
                for folder_name in ["./snapshots/"+file_path_customer, "./snapshots/"+file_path_employee]:
                    for root, dirs, files in os.walk(folder_name):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, os.path.join(folder_name, '..'))
                            zip_file.write(file_path, arcname)
        background_tasks.add_task(os.remove, f'./{zipfile_name}')
        return FileResponse(f'./{zipfile_name}', media_type='application/zip', filename=zipfile_name)
    except Exception as e:
        return JSONResponse(content={
            'status': 2,
            'message': str(e)
        })
        
@app.post("/recover_db", tags=["Database"],
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
    try:
        if not file.filename.endswith('.zip'):
            raise HTTPException(status_code=400, detail="Invalid file format. Please upload a zip file.")

        temp_zip_path = f"./snapshots/{file.filename}"

        with open(temp_zip_path, "wb") as buffer:
            buffer.write(await file.read())
        
        extract_name = f"extracted_{os.path.splitext(file.filename)[0]}"
        extract_dir = f"snapshots/{extract_name}"

        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        os.remove(temp_zip_path)

        extracted_files = []
        folders = os.listdir(extract_dir)
        for root, dirs, files in os.walk(extract_dir):
            for name in files:
                extracted_files.append(os.path.join(root, name))
        for folder in folders:
            # print(folder)
            for snapshot_name in os.listdir(os.path.join(extract_dir, folder)):
                if snapshot_name.endswith('.snapshot'):
                    # print(os.path.join(extract_name, folder, snapshot_name))
                    json_post = {
                        "collection_name": folder,
                        "snapshot_name": os.path.join(extract_name, folder, snapshot_name)
                        }
                    check = requests.post(URL_RECOVER_SNAP, json=json_post)
                    
                    if check.status_code != 200:
                        return JSONResponse(content={
                            'status': 2,
                            # 'message': f"Error when recover database {folder}"
                            'message': check.json()['message']
                        })  
        try:
            shutil.rmtree(extract_dir)
            del extracted_files, folders, folder, snapshot_name, json_post, check
            gc.collect()
            return JSONResponse(content={
                'status': 1,
                'message': "Recover database successfully"
            })
        except:
            return JSONResponse(content={
                'status': 1,
                'message': "Recover database successfully"
            })
    except Exception as e:
        return JSONResponse(content={
            'status': 2,
            'message': str(e)
        })

#if __name__ == "__main__":
#     s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#     s.connect(("8.8.8.8", 80))
#    ip_address = s.getsockname()[0]
#     s.close()
    
#    uvicorn.run(app, host=ip_address, port=2011)
