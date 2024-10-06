from fastapi import FastAPI, File, UploadFile, status, Query, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from models.yolo import YOLOv8_face
from pydantic import BaseModel
from deepface import DeepFace
from scipy.spatial import distance as dist

import cv2
import numpy as np
import mediapipe as mp
import math
import requests
import zipfile
import os
import base64
import datetime
import shutil
import gc


modelpath ='./models/yolov8n-face.onnx'
confThreshold = 0.8
nmsThreshold = 0.7
YOLOv8_face_detector = YOLOv8_face(modelpath, conf_thres=confThreshold, iou_thres=nmsThreshold)

# Khởi tạo Mediapipe
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
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

FastDB_HOST = os.getenv("FASTAPI_HOST", "localhost")
FastDB_PORT = int( os.getenv("FASTAPI_PORT", "7005"))

ip_private = f'http://{FastDB_HOST}:{FastDB_PORT}'
URL_SEARCH = f'{ip_private}/search_point'
URL_INSERT = f'{ip_private}/insert_point'
URL_DELETE = f'{ip_private}/delete_point'
URL_RECOVER_SNAP = f'{ip_private}/recover_snapshot'
URL_CREATE_SNAP = f'{ip_private}/create_snapshot'
URL_GET_CLT = f'{ip_private}/get_collections'
URL_CRE_CLT = f'{ip_private}/create_collection'

KNOWN_FACE_WIDTH = 14.3  # centimeter
# Indices của các điểm đặc trưng trên mắt trong Mediapipe Face Mesh
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

# Ngưỡng để xác định mắt đang nhắm
EYE_AR_THRESH = 0.25

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
    img_base64: str = Query(None, description="Ảnh chứa mặt để đăng ký")
    id: str = Query(None, description="ID của khách hàng")
    name: str = Query(None, description="Tên của khách hàng")
    role: str = Query(None, description="1: Nhân viên, 0: Khách hàng")
    store_id: str = Query(None, description="ID cửa hàng")


class DeleteFace(BaseModel):
    id: str = Query(None, description="ID của khách hàng")
    store_id: str = Query(None, description="ID cửa hàng")
    # role: str = Query(None, description="1: Nhân viên, 0: Khách hàng")


class FaceRecog(BaseModel):
    img_base64: str = Query(None, description="Ảnh chứa mặt để nhận diện")
    role: str = Query(None, description="1: Nhân viên, 0: Khách hàng")
    store_id: str = Query(None, description="ID cửa hàng")


def get_embedding(imgf,imgf_real):
    embedding_objs = DeepFace.represent(
        img_path = imgf,
        model_name= "VGG-Face",
        detector_backend = "skip",
        align = True,
        normalization = "VGGFace2",
        anti_spoofing = True,
    )
    face_is_real = DeepFace.extract_faces(
        img_path = imgf_real,
        detector_backend = "yolov8",
        align = True,
        anti_spoofing = True,
    )
    # get confidence largest
    index_confidence_face = 0
    max_confidence = 0
    # nmsThreshold = 0.7
    if len(face_is_real) > 1:
        for i in range(len(face_is_real)):
            if face_is_real[i]['confidence'] > max_confidence:
                max_confidence = face_is_real[i]['confidence']
                index_confidence_face = i
    return embedding_objs[0]['embedding'],face_is_real[index_confidence_face]["is_real"]

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def save_face_image(data, face,id,name,is_checkin=True):
    if is_checkin:
        if data.role == '0':
            folder_save = "data_face_checkin_customer_images"
        else:
            folder_save = "data_face_checkin_employee_images"
    else:
        if data.role == '0':
            folder_save = "data_face_register_customer_images"
        else:
            folder_save = "data_face_register_employee_images"
    # create folder to save face
    if not os.path.exists(f'./{folder_save}'):
        os.makedirs(f'./{folder_save}')
    
    if not os.path.exists(f'./{folder_save}/{data.store_id}'):
        os.makedirs(f'./{folder_save}/{data.store_id}')
    
    time_checkin = datetime.datetime.now().strftime("%Y_%m_%d")
    
    if not os.path.exists(f'./{folder_save}/{data.store_id}/{time_checkin}'):
        os.makedirs(f'./{folder_save}/{data.store_id}/{time_checkin}')
    
    second_checkin = datetime.datetime.now().strftime("%H_%M_%S")
    cv2.imwrite(f'./{folder_save}/{data.store_id}/{time_checkin}/{id}_{name}_{second_checkin}.jpg', face)

def distance_face_to_camera(bbox_face, width_or):
    xmin, ymin, xmax, ymax = bbox_face
    P = xmax - xmin
    Fmm = 4
    width = width_or
    F_pixel = (Fmm * width) / 4.8 # 4.8 is the width of the mobile phone camera sensor in mm
    # F_pixel = focal_length_value
    W_face = KNOWN_FACE_WIDTH
    D = (W_face * F_pixel) / P
    return D

def check_detect_blur(img, threshold=350):
    # Đọc hình ảnh
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Tính toán biến thiên của Laplacian
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    print("laplacian_var", laplacian_var)
    # Kiểm tra nếu giá trị biến thiên nhỏ hơn ngưỡng (threshold)
    if laplacian_var < threshold:
        return False
    else:
        return True

def eye_aspect_ratio(eye_landmarks, face_landmarks):
    A = dist.euclidean([face_landmarks[eye_landmarks[1]].x, face_landmarks[eye_landmarks[1]].y],
                       [face_landmarks[eye_landmarks[5]].x, face_landmarks[eye_landmarks[5]].y])
    B = dist.euclidean([face_landmarks[eye_landmarks[2]].x, face_landmarks[eye_landmarks[2]].y],
                       [face_landmarks[eye_landmarks[4]].x, face_landmarks[eye_landmarks[4]].y])
    C = dist.euclidean([face_landmarks[eye_landmarks[0]].x, face_landmarks[eye_landmarks[0]].y],
                       [face_landmarks[eye_landmarks[3]].x, face_landmarks[eye_landmarks[3]].y])
    ear = (A + B) / (2.0 * C)
    return ear

def check_condition(data, is_checkin=True):
    if is_checkin == False:
        if data.id is None or data.name is None or data.id == "" or data.name == "":
            return JSONResponse(content={
                'status': 2,
                'message': "id and name are required"
            })
    
    if len(data.img_base64) == 0:
        return JSONResponse(content={
            'status': 2,
            'message': "img_base64 is required"
        })
    
    if data.role != '1' and data.role != '0':
        return JSONResponse(content={
            'status': 2,
            'message': "role is 0 or 1"
        })
    
    if data.store_id is None or data.store_id == "":
        return JSONResponse(content={
            'status': 2,
            'message': "store_id is required"
        })
    return True

def check_eyes_open(img_decode):
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        frame_rgb = cv2.cvtColor(img_decode, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_ear = eye_aspect_ratio(LEFT_EYE_LANDMARKS, face_landmarks.landmark)
                right_ear = eye_aspect_ratio(RIGHT_EYE_LANDMARKS, face_landmarks.landmark)
                ear = (left_ear + right_ear) / 2.0
                if ear < EYE_AR_THRESH:
                    return False
                else:
                    return True
                
def ConvertToPoint(landmark):
    return [landmark.x, landmark.y, landmark.z]

def CalcDistance(point1, point2):
    x1, y1, z1 = ConvertToPoint(point1)
    x2, y2, z2 = ConvertToPoint(point2)
    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return distance

def DetectDirection(landmark):
    left = CalcDistance(landmark[5], landmark[234])
    right = CalcDistance(landmark[5], landmark[454])

    threshold = 2.5
    result = "straight"

    if(left < right):
        ratio = right / left
        if(ratio > threshold):
            result = "right"
    elif(right < left):
        ratio = left / right
        if(ratio > threshold):
            result = "left"
    
    return result

def check_face_left_right(img_decode):
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img_decode, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return False, "Face not detected! Please try again"

        landmarks = results.multi_face_landmarks
        if(len(landmarks) == 0):
            return False, "Face not detected! Please try again"
        landmark = landmarks[0].landmark    
        direction = DetectDirection(landmark)
        if direction == "left":
            return False, "Face is looking left! Please look straight"
        elif direction == "right":
            return False, "Face is looking right! Please look straight"
        else:
            return True, "Face is looking straight"

def is_full_face(image):
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        height, width = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Phát hiện khuôn mặt
        results = face_detection.process(image_rgb)
        # Kiểm tra từng khuôn mặt
        if results.detections:
            for detection in results.detections:
                face_landmarks = detection.location_data.relative_keypoints
                eye_left = face_landmarks[1]
                eye_right = face_landmarks[0]
                ears_right = face_landmarks[4]
                ears_left = face_landmarks[5]
                noise = face_landmarks[2]
                mouth = face_landmarks[3]
                
                x_mouth = (mouth.x * image.shape[1])
                y_mouth = (mouth.y * image.shape[0])
                
                x_ears_right = (ears_right.x * image.shape[1])
                y_ears_right = (ears_right.y * image.shape[0])
                
                x_ears_left = (ears_left.x * image.shape[1])
                y_ears_left = (ears_left.y * image.shape[0])
                
                x_eye_left = (eye_left.x * image.shape[1])
                y_eye_left = (eye_left.y * image.shape[0])
                
                x_eye_right = (eye_right.x * image.shape[1])
                y_eye_right = (eye_right.y * image.shape[0])
                
                x_noise = (noise.x * image.shape[1])
                y_noise = (noise.y * image.shape[0])
                
                if x_mouth > width or y_mouth > height:
                    # print("Mouth not in size face")
                    return False, "Your mouth is not detected! Please show your face"
                    
                if x_ears_right > width or y_ears_right > height:
                    return False, "Your right ear is not detected! Please show your face"

                    
                if x_ears_left > width or y_ears_left > height:
                    return False, "Your left ear is not detected! Please show your face"
                    
                if x_eye_left > width or y_eye_left > height:
                    return False, "Your left eye is not detected! Please show your face"
                    
                if x_eye_right > width or y_eye_right > height:
                    return False, "Your right eye is not detected! Please show your face"
            
                if x_noise > width or y_noise > height:
                    return False, "Your noise is not detected! Please show your face"
            return True, "Face is detected"
        else:
            return False, "Face is not detected"

def detect_n_emb_face(data):
    try:
        # print("id", data.id)
        # print("name", data.name)
        # print("store_id", data.store_id)
        contents = data.img_base64
        contents = base64.b64decode(contents)
        img_decode = cv2.imdecode(np.frombuffer(contents, np.uint8), -1)
        
        check_flr, message_flr = check_face_left_right(img_decode)
        print("check_flr", check_flr)
        if check_flr == False:
            return False,JSONResponse(content={
                'status': 2,
                'message': message_flr
            })
        
        check_eyes = check_eyes_open(img_decode)
        print("check_eyes", check_eyes)
        if check_eyes == False:
            return False,JSONResponse(content={
                'status': 2,
                'message': "Eyes are closed! Please open your eyes"
            })
        boxes, scores, classIds, kpts = YOLOv8_face_detector.detect(img_decode)
        print("Scores", scores)
    except Exception as e:
        del img_decode, contents
        gc.collect()
        return False,JSONResponse(content={
            'status': 2,
            'message': "Error when detecting face! Please try again"
        })
    idx_large = np.argmax(scores)
    box = boxes[idx_large]
    x,y,w,h = box
    x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
    # mở rộng khuôn mặt ra 10px 
    x1 = x1 - 10 if x1 - 10 > 0 else 0
    y1 = y1 - 10 if y1 - 10 > 0 else 0
    x2 = x2 + 10 if x2 + 10 < img_decode.shape[1] else img_decode.shape[1]
    y2 = y2 + 10 if y2 + 10 < img_decode.shape[0] else img_decode.shape[0]
    
    distance = distance_face_to_camera((x1, y1, x2, y2), img_decode.shape[1])
    print("distance", distance)
    if distance < 30 or distance > 70:
        return False,JSONResponse(content={
            'status': 2,
            'message': "Face is too close or too far! Please move back or forward"
        })
    
    face = img_decode[y1:y2, x1:x2]
    face = face.astype('uint8')
    
    check_full_face,mess_full_face = is_full_face(face)
    print("check_full_face", check_full_face)
    if check_full_face == False:
        return False,JSONResponse(content={
            'status': 2,
            'message': mess_full_face
        })
    
    check_face_blur = check_detect_blur(face)
    print("check_face_blur", check_face_blur)
    if check_face_blur == False:
        return False,JSONResponse(content={
            'status': 2,
            'message': "Face is blur! Please keep your face in focus"
        })
    
    face = adjust_gamma(face, gamma=1.5)

    try:
        emb,is_real = get_embedding(face, img_decode)
        if is_real == False:
            return False,JSONResponse(content={
                'status': 2,
                'message': "Face is not real! Please use your real face"
            })
    except Exception as e:
        del face, img_decode
        gc.collect()
        return False,JSONResponse(content={
            'status': 2,
            # 'message': "Error when encoding face"
            "message": "Error! Please try again"
        })
    return True, (emb, img_decode)

def cnc_clt_exist(store_id):
    headers = {
        'Content-Type': 'application/json',
    }
    check_clt = requests.get(URL_GET_CLT, headers=headers).json()['collections']
    if f"{store_id}_Employees" not in check_clt and f"{store_id}_Customers" not in check_clt:
        print("Create collection")
        data_cus = {
            "collection_name": f"{store_id}_Customers"
        }
        data_emp = {
            "collection_name": f"{store_id}_Employees"
        }
        result_cus = requests.post(URL_CRE_CLT, json=data_cus)
        result_emp = requests.post(URL_CRE_CLT, json=data_emp)
        if result_cus.status_code != 201 or result_emp.status_code != 201:
            return False
    return True

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

@app.get("/check_connection", description="Check connection")
async def check_connection():
    try:
        image = cv2.imread('face_fake_new.png')
        
        # img_path=img_path,
        #     detector_backend=detector_backend,
        #     grayscale=False,
        #     enforce_detection=enforce_detection,
        #     align=align,
        #     expand_percentage=expand_percentage,
        #     anti_spoofing=anti_spoofing,
        #     max_faces=max_faces,
        
        face_is_real = DeepFace.extract_faces(
            img_path = image,
            detector_backend = "yolov8",
            align = True,
            anti_spoofing = True,
        )
        print(face_is_real)
        return face_is_real[0]["is_real"]
    except Exception as e:
        print(e)
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
    
    check_condition_face = check_condition(data, is_checkin=True)
    if check_condition_face != True:
        return check_condition_face
    
    if data.role == '1':
        collection_name=f'{data.store_id}_Employees'
    elif data.role == '0':
        collection_name=f'{data.store_id}_Customers'

    check_emb, message = detect_n_emb_face(data)
    if check_emb == False:
        # print(message)
        return message
    
    emb, img_decode = message
    
    result_cnc_clt = cnc_clt_exist(data.store_id)
    if result_cnc_clt == False:
        return JSONResponse(content={
            'status': 2,
            'message': "Error when create collection"
        })
    data_search = {
        "collection_name": collection_name,
        "vector_embedding": emb,
        "store_id": data.store_id
    }
    print(requests.post(URL_SEARCH, json=data_search).json())
    search_db = requests.post(URL_SEARCH, json=data_search).json()['data']
    search_db = search_db[0] if len(search_db) > 0 else []
    name = search_db[1]['name'] if len(search_db) > 0 else "Unknown"
    id = search_db[1]['id'] if len(search_db) > 0 else "Unknown"
    time_created = search_db[1]['time_created'] if len(search_db) > 0 else "Unknown"
    if id == "Unknown" and name == "Unknown":
        return JSONResponse(content={
            'status': 0,
            'message': "Face is not existed! Please register your face or checkin again"
        })
    print({
            'status': 1,
            'id': id,
            'name': name,
        })
    save_face_image(data, img_decode, id, name)
    return JSONResponse(content={
        'status': 1,
        'id': id,
        'name': name,
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
        Create a face from an image base64.

        Parameters:
            - data (CreateFace): The data containing the image base64, id, name, and role.

        Returns:
            - JSONResponse: The response containing the status and message of the face creation process.
                - status (int): The status code of the response.
                - message (str): The message indicating the result of the face creation process.
    """
    id = data.id
    name = data.name
    store_id = data.store_id
    print("id", id)
    print("name", name)
    print("store_id", store_id)
    check_condition_face = check_condition(data, is_checkin=False)
    if check_condition_face != True:
        return check_condition_face

    if data.role == '1':
        collection_name=f'{store_id}_Employees'
    elif data.role == '0':
        collection_name=f'{store_id}_Customers'
    
    check_emb, message = detect_n_emb_face(data)
    if check_emb == False:
        # print(message)
        return message
    
    emb, img_decode = message

    result_cnc_clt = cnc_clt_exist(data.store_id)
    if result_cnc_clt == False:
        return JSONResponse(content={
            'status': 2,
            # 'message': "Error when create collection"
            'message': "Error"
        })
    
    # check if id is existed
    data_search = {
        "collection_name": collection_name,
        "vector_embedding": emb,
        "store_id": data.store_id
    }
    # print(requests.post(URL_SEARCH, json=data_search))
    search_db = requests.post(URL_SEARCH, json=data_search).json()['data']
    search_db = search_db[0] if len(search_db) > 0 else []
    if len(search_db) > 0:
        return JSONResponse(content={
            'status': 0,
            'message': f'id {id} is existed'
        })
    data_insert = {
            "collection_name": collection_name,
            "vector_embedding": emb,
            "id": id,
            "name": name,
            "store_id": store_id
        }
    check = requests.post(URL_INSERT, json=data_insert)
    if check.status_code != 201:
        return JSONResponse(content={
            'status': 2,
            'message': "Error when insert face"
        })
    save_face_image(data, img_decode, id, name, is_checkin=False)
    # del search_db, emb, face, img_decode
    # gc.collect()
    return JSONResponse(content={
        'status': 1,
        'message': f'Create face {name} with id {id} successfully'
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
        return JSONResponse(content={
            'status': 0,
            'message': "Not found employee with id {id_em}"
        })
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
            tags=["Database"]
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
        if clt.endswith('Customers'):
            files_path_customer.append(clt)
        elif clt.endswith('Employees'):
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
