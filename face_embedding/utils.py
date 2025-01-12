from deepface import DeepFace
from typing import *
from dotenv import load_dotenv
from scipy.spatial import distance as dist

import numpy as np
import cv2
import os
import datetime
import mediapipe as mp
import math
import requests


load_dotenv(dotenv_path=".env")

FastDB_HOST = os.getenv("FASTAPI_HOST")
FastDB_PORT = int(os.getenv("FASTAPI_PORT"))

ip_private = f'http://{FastDB_HOST}:{FastDB_PORT}'
# URL_SEARCH = os.getenv("URL_SEARCH").format(ip_private = ip_private)
# URL_INSERT = os.getenv("URL_INSERT").format(ip_private = ip_private)
# URL_DELETE = os.getenv("URL_DELETE").format(ip_private = ip_private)
# URL_RECOVER_SNAP = os.getenv("URL_RECOVER_SNAP").format(ip_private = ip_private)
# URL_CREATE_SNAP = os.getenv("URL_CREATE_SNAP").format(ip_private = ip_private)
URL_GET_CLT = os.getenv("URL_GET_CLT").format(ip_private = ip_private)
URL_CRE_CLT = os.getenv("URL_CRE_CLT").format(ip_private = ip_private)


KNOWN_FACE_WIDTH = float(os.getenv("KNOWN_FACE_WIDTH"))
# Indices của các điểm đặc trưng trên mắt trong Mediapipe Face Mesh
LEFT_EYE_LANDMARKS = list(map(int, os.getenv("LEFT_EYE_LANDMARKS").strip('[]').split(', ')))
RIGHT_EYE_LANDMARKS = list(map(int, os.getenv("RIGHT_EYE_LANDMARKS").strip('[]').split(', ')))

# Ngưỡng để xác định mắt đang nhắm
EYE_AR_THRESH = float(os.getenv("EYE_AR_THRESH"))

BLUR_THRESHOLD = int(os.getenv("BLUR_THRESHOLD"))

LEFT_RIGHT_FACE_THRESHOLD = float(os.getenv("LEFT_RIGHT_FACE_THRESHOLD"))

# Khởi tạo Mediapipe
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

def get_embedding(imgf,imgf_real):
    """
    Get embedding from ndarray image and check face is real or not
    """
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
    '''
    Tăng độ sáng của khuôn mặt
    '''
    
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def save_face_image(data, face,id,name,is_checkin=True):
    '''
    Lưu ảnh khuôn mặt vào thư mục tương ứng
    '''
    if is_checkin:
        if data.role == '0':
            folder_save = os.getenv("CHECKIN_CUSTOMER_PATH")
        else:
            folder_save = os.getenv("CHECKIN_EMPLOYEE_PATH")
    else:
        if data.role == '0':
            folder_save = os.getenv("REGISTER_CUSTOMER_PATH")
        else:
            folder_save = os.getenv("REGISTER_EMPLOYEE_PATH")
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
    
def distance_face_to_camera(bbox_face, width_or) -> float:
    '''
    Tính khoảng cách từ khuôn mặt đến camera
    '''
    
    xmin, ymin, xmax, ymax = bbox_face
    P = xmax - xmin
    Fmm = 4
    width = width_or
    F_pixel = (Fmm * width) / 4.8 # 4.8 is the width of the mobile phone camera sensor in mm
    # F_pixel = focal_length_value
    W_face = KNOWN_FACE_WIDTH
    D = (W_face * F_pixel) / P
    return D

def check_detect_blur(img, threshold=BLUR_THRESHOLD):
    '''
    Kiểm tra xem hình ảnh có bị mờ hay không
    '''
    
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
    '''
    Tính tỉ lệ khả năng mở mắt
    '''
    
    A = dist.euclidean([face_landmarks[eye_landmarks[1]].x, face_landmarks[eye_landmarks[1]].y],
                    [face_landmarks[eye_landmarks[5]].x, face_landmarks[eye_landmarks[5]].y])
    B = dist.euclidean([face_landmarks[eye_landmarks[2]].x, face_landmarks[eye_landmarks[2]].y],
                    [face_landmarks[eye_landmarks[4]].x, face_landmarks[eye_landmarks[4]].y])
    C = dist.euclidean([face_landmarks[eye_landmarks[0]].x, face_landmarks[eye_landmarks[0]].y],
                    [face_landmarks[eye_landmarks[3]].x, face_landmarks[eye_landmarks[3]].y])
    ear = (A + B) / (2.0 * C)
    return ear

def check_eyes_open(img_decode):
    '''
    Kiểm tra xem mắt có đang mở hay không
    '''
    
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

def DetectDirection(landmark, threshold=LEFT_RIGHT_FACE_THRESHOLD):
    left = CalcDistance(landmark[5], landmark[234])
    right = CalcDistance(landmark[5], landmark[454])

    # threshold = 2.5
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
    '''
    Kiểm tra xem khuôn mặt có nhìn thẳng vào camera hay không
    '''
    
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
    '''
    Kiểm tra xem khuôn mặt có đầy đủ các điểm đặc trưng hay không
    '''
    
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
                # ears_right = face_landmarks[4]
                # ears_left = face_landmarks[5]
                noise = face_landmarks[2]
                mouth = face_landmarks[3]
                
                x_mouth = (mouth.x * image.shape[1])
                y_mouth = (mouth.y * image.shape[0])
                
                # x_ears_right = (ears_right.x * image.shape[1])
                # y_ears_right = (ears_right.y * image.shape[0])
                
                # x_ears_left = (ears_left.x * image.shape[1])
                # y_ears_left = (ears_left.y * image.shape[0])
                
                x_eye_left = (eye_left.x * image.shape[1])
                y_eye_left = (eye_left.y * image.shape[0])
                
                x_eye_right = (eye_right.x * image.shape[1])
                y_eye_right = (eye_right.y * image.shape[0])
                
                x_noise = (noise.x * image.shape[1])
                y_noise = (noise.y * image.shape[0])
                
                if x_mouth > width or y_mouth > height:
                    # print("Mouth not in size face")
                    return False, "Your mouth is not detected! Please show your face"
                    
                # if x_ears_right > width or y_ears_right > height:
                #     return False, "Your right ear is not detected! Please show your face"

                    
                # if x_ears_left > width or y_ears_left > height:
                #     return False, "Your left ear is not detected! Please show your face"
                    
                if x_eye_left > width or y_eye_left > height:
                    return False, "Your left eye is not detected! Please show your face"
                    
                if x_eye_right > width or y_eye_right > height:
                    return False, "Your right eye is not detected! Please show your face"
            
                if x_noise > width or y_noise > height:
                    return False, "Your noise is not detected! Please show your face"
            return True, "Face is detected"
        else:
            return False, "Face is not detected"


def check_face_mask(model, img_decode, box):
    '''
    Kiểm tra xem khuôn mặt có đeo khẩu trang hay không
    '''
    x,y,w,h = box
    x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
    # mở rộng khuôn mặt ra FACE_EXTpx 
    x1 = x1 - 80 if x1 - 80 > 0 else 0
    y1 = y1 - 80 if y1 - 80 > 0 else 0
    x2 = x2 + 80 if x2 + 80 < img_decode.shape[1] else img_decode.shape[1]
    y2 = y2 + 80 if y2 + 80 < img_decode.shape[0] else img_decode.shape[0]
    
    face = img_decode[y1:y2, x1:x2]
    face = face.astype('uint8')
    
    # save face image
    # cv2.imwrite("face.jpg", face)
    
    try:    
        prediction = model.predict(face)
        # with open("mask_detection.txt", "w") as f:
        #     # f.write(str(prediction.boxes))
        #     for result in prediction:
        #         f.write(str(result.boxes))
        labels = prediction[0]
        # print(labels)
        class_id = int(prediction[0].boxes[0].cls)

        if class_id == 1 or class_id == 2:
            return False, "Your face is wearing a mask! Please remove the mask"
    except:
        return False, "Please checkin again!"
    return True, "Face is not wearing a mask"

def cnc_clt_exist(store_id):
    '''
    Kiểm tra xem collection đã tồn tại chưa và tạo collection nếu chưa tồn tại
    '''
    
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

def check_condition(data, is_checkin=True):
    '''
    Kiểm tra điều kiện để thực hiện checkin hoặc đăng ký
    '''
    
    if is_checkin == False:
        if data.id is None or data.name is None or data.id == "" or data.name == "":
            return False, "id and name are required"
    
    if len(data.img_base64) == 0:
        # return JSONResponse(content={
        #     'status': 2,
        #     'message': "img_base64 is required"
        # })
        return False, "invalid"
    
    if data.role != '1' and data.role != '0':
        # return JSONResponse(content={
        #     'status': 2,
        #     'message': "role is 0 or 1"
        # })
        return False, "invalid"
    
    if data.store_id is None or data.store_id == "":
        # return JSONResponse(content={
        #     'status': 2,
        #     'message': "store_id is required"
        # })
        return False, "store_id is required"
    return True, "Success"