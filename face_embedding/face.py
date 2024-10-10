from fastapi import FastAPI, File, UploadFile, Query, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from models.yolo import YOLOv8_face
from pydantic import BaseModel
from deepface import DeepFace
from ultralytics import YOLO

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

load_dotenv(dotenv_path=".env")

modelpath ='./models/yolov8n-face.onnx'
confThreshold = 0.8
nmsThreshold = 0.7
YOLOv8_face_detector = YOLOv8_face(modelpath, conf_thres=confThreshold, iou_thres=nmsThreshold)

model_face_mask = YOLO("./models/yolov8n-facemask.pt")


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
    is_update: str = Query("0", description="1: Update face, 0: Create face")


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
        # print("img_decode", img_decode.shape)
        
        if is_checkin == True:
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

        if is_detect_face == True:
            boxes, scores, classIds, kpts = YOLOv8_face_detector.detect(img_decode)
            print("Scores", scores)
        else:
            scores = [0.9]
            img_size = img_decode.shape
            boxes = [[0, 0, img_size[1], img_size[0]]]
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
    # mở rộng khuôn mặt ra FACE_EXTpx 
    x1 = x1 - FACE_EXT if x1 - FACE_EXT > 0 else 0
    y1 = y1 - FACE_EXT if y1 - FACE_EXT > 0 else 0
    x2 = x2 + FACE_EXT if x2 + FACE_EXT < img_decode.shape[1] else img_decode.shape[1]
    y2 = y2 + FACE_EXT if y2 + FACE_EXT < img_decode.shape[0] else img_decode.shape[0]
    
    if is_checkin == True:
        distance = distance_face_to_camera((x1, y1, x2, y2), img_decode.shape[1])
        print("distance", distance)
        if distance < 30 or distance > 70:
            return False,JSONResponse(content={
                'status': 2,
                'message': "Face is too close or too far! Please move back or forward"
            })
    
    face = img_decode[y1:y2, x1:x2]
    face = face.astype('uint8')
    
    if is_checkin == True:
        check_face_is_mask, message_face_is_mask = check_face_mask(model_face_mask, face)
        
        if check_face_is_mask == False:
            return False,JSONResponse(content={
                'status': 2,
                'message': message_face_is_mask
            })
        
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
        if is_real == False and is_checkin == True:
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
    check_condition_face, message_condition_face = check_condition(data, is_checkin=True)
    if check_condition_face == False:
        return JSONResponse(content={
            'status': 2,
            'message': message_condition_face
        })
    
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
    
    # print(requests.post(URL_SEARCH, json=data_search).json())
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
    check_condition_face, message_condition_face = check_condition(data, is_checkin=True)
    if check_condition_face == False:
        return JSONResponse(content={
            'status': 2,
            'message': message_condition_face
        })
    

    if data.role == '1':
        collection_name=f'{store_id}_Employees'
    elif data.role == '0':
        collection_name=f'{store_id}_Customers'
    
    check_emb, message = detect_n_emb_face(data, is_checkin=False)
    if check_emb == False:
        return message
    
    emb, img_decode = message

    result_cnc_clt = cnc_clt_exist(data.store_id)
    if result_cnc_clt == False:
        return JSONResponse(content={
            'status': 2,
            # 'message': "Error when create collection"
            'message': "Error! Please try again"
        })
    
    # check if id is existed
    data_search = {
        "collection_name": collection_name,
        "vector_embedding": emb,
        "store_id": data.store_id
    }
    # print(data_search)
    if data.is_update == '0':
        print(requests.post(URL_SEARCH, json=data_search).json())
        search_db = requests.post(URL_SEARCH, json=data_search).json()['data']
        search_db = search_db[0] if len(search_db) > 0 else []

        if len(search_db) > 0:
            return JSONResponse(content={
                'status': 0,
                # 'message': f'id {id} is existed'
                'message': "Face is existed! Please use another face"
            })

        data_insert = {
                "collection_name": collection_name,
                "vector_embedding": emb,
                "id": id,
                "name": name,
                "store_id": store_id
            }
    else:
        print("Update face")
        data_insert = {
                "collection_name": collection_name,
                "vector_embedding": emb,
                "id": id,
                "name": name,
                "store_id": store_id,
                "is_update_id": "true"
            }

    check = requests.post(URL_INSERT, json=data_insert)
    if check.status_code != 201:
        return JSONResponse(content={
            'status': 2,
            'message': "Error when insert face"
        })
    
    save_face_image(data, img_decode, id, name, is_checkin=False)
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
            'message': f"Not found employee with id {id_em}"
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
