"""
Legacy utilities imported from original utils.py for backward compatibility.
This file contains utility functions from the original codebase.
"""
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
import logging

from io import BytesIO

# Import from new structure
from config.settings import get_settings

settings = get_settings()

# Legacy variables for backward compatibility
FastDB_HOST = settings.QDRANT_DB_HOST
FastDB_PORT = settings.QDRANT_DB_PORT

ip_private = f'http://{FastDB_HOST}:{FastDB_PORT}'

KNOWN_FACE_WIDTH = settings.KNOWN_FACE_WIDTH
LEFT_EYE_LANDMARKS = eval(settings.LEFT_EYE_LANDMARKS)
RIGHT_EYE_LANDMARKS = eval(settings.RIGHT_EYE_LANDMARKS)
EYE_AR_THRESH = settings.EYE_AR_THRESH
BLUR_THRESHOLD = settings.BLUR_THRESHOLD
LEFT_RIGHT_FACE_THRESHOLD = settings.LEFT_RIGHT_FACE_THRESHOLD
FACE_EXT = settings.FACE_EXT
CONF_THRESHOLD = settings.CONF_THRESHOLD

# Initialize Mediapipe
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

logger = logging.getLogger(__name__)


def get_embedding(imgf, imgf_real):
    """
    Get embedding from ndarray image and check face is real or not
    """
    embedding_objs = DeepFace.represent(
        img_path=imgf,
        model_name="VGG-Face",
        detector_backend="skip",
        align=True,
        normalization="VGGFace2",
        anti_spoofing=True,
    )
    face_is_real = DeepFace.extract_faces(
        img_path=imgf_real,
        detector_backend="yolov8",
        align=True,
        anti_spoofing=True,
    )
    # get confidence largest
    index_confidence_face = 0
    max_confidence = 0
    if len(face_is_real) > 1:
        for i in range(len(face_is_real)):
            if face_is_real[i]['confidence'] > max_confidence:
                max_confidence = face_is_real[i]['confidence']
                index_confidence_face = i
    return embedding_objs[0]['embedding'], face_is_real[index_confidence_face]["is_real"]


def detect_face(image):
    """Detect faces in image using DeepFace YOLO backend."""
    boxes, scores, distances = [], [], []
    face_detected = DeepFace.extract_faces(
        img_path=image,
        detector_backend="yolov8",
        align=True,
        expand_percentage=FACE_EXT,
        anti_spoofing=True,
    )
    
    for i in range(len(face_detected)):
        score = face_detected[i]['confidence']
        spoofing = face_detected[i]['is_real']
        
        if score < CONF_THRESHOLD or not spoofing:
            break
        scores.append(score)
        x, y, w, h, le, re = face_detected[i]['facial_area'].values()
        xmin, ymin, xmax, ymax = x, y, x+w, y+h
    
        distance = distance_face_to_camera((xmin, ymin, xmax, ymax), image.shape[1])
        
        distances.append(distance)
        boxes.append([x, y, w, h])
    
    return boxes, np.array(scores), np.array(distances)


def adjust_gamma(image, gamma=1.0):
    """Adjust brightness of face image."""
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def distance_face_to_camera(bbox_face, width_or) -> float:
    """Calculate distance from face to camera."""
    xmin, ymin, xmax, ymax = bbox_face
    P = xmax - xmin
    Fmm = 4
    width = width_or
    F_pixel = (Fmm * width) / 4.8  # 4.8 is the width of the mobile phone camera sensor in mm
    W_face = KNOWN_FACE_WIDTH
    D = (W_face * F_pixel) / P
    return D


def check_detect_blur(img, threshold=BLUR_THRESHOLD):
    """Check if image is blurred."""
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian_var >= threshold


def eye_aspect_ratio(eye_landmarks, face_landmarks):
    """Calculate eye aspect ratio."""
    A = dist.euclidean([face_landmarks[eye_landmarks[1]].x, face_landmarks[eye_landmarks[1]].y],
                      [face_landmarks[eye_landmarks[5]].x, face_landmarks[eye_landmarks[5]].y])
    B = dist.euclidean([face_landmarks[eye_landmarks[2]].x, face_landmarks[eye_landmarks[2]].y],
                      [face_landmarks[eye_landmarks[4]].x, face_landmarks[eye_landmarks[4]].y])
    C = dist.euclidean([face_landmarks[eye_landmarks[0]].x, face_landmarks[eye_landmarks[0]].y],
                      [face_landmarks[eye_landmarks[3]].x, face_landmarks[eye_landmarks[3]].y])
    ear = (A + B) / (2.0 * C)
    return ear


def check_eyes_open(img_decode):
    """Check if eyes are open."""
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        frame_rgb = cv2.cvtColor(img_decode, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_ear = eye_aspect_ratio(LEFT_EYE_LANDMARKS, face_landmarks.landmark)
                right_ear = eye_aspect_ratio(RIGHT_EYE_LANDMARKS, face_landmarks.landmark)
                ear = (left_ear + right_ear) / 2.0
                return ear >= EYE_AR_THRESH
    return False


def ConvertToPoint(landmark):
    """Convert landmark to point."""
    return [landmark.x, landmark.y, landmark.z]


def CalcDistance(point1, point2):
    """Calculate distance between two points."""
    x1, y1, z1 = ConvertToPoint(point1)
    x2, y2, z2 = ConvertToPoint(point2)
    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return distance


def DetectDirection(landmark, threshold=LEFT_RIGHT_FACE_THRESHOLD):
    """Detect face direction."""
    left = CalcDistance(landmark[5], landmark[234])
    right = CalcDistance(landmark[5], landmark[454])

    result = "straight"

    if left < right:
        ratio = right / left
        if ratio > threshold:
            result = "right"
    elif right < left:
        ratio = left / right
        if ratio > threshold:
            result = "left"
    
    return result


def check_face_left_right(img_decode):
    """Check if face is looking straight."""
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img_decode, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return False, "Face not detected! Please try again"

        landmarks = results.multi_face_landmarks
        if len(landmarks) == 0:
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
    """Check if face has all required features."""
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        height, width = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)
        
        if results.detections:
            for detection in results.detections:
                face_landmarks = detection.location_data.relative_keypoints
                eye_left = face_landmarks[1]
                eye_right = face_landmarks[0]
                noise = face_landmarks[2]
                mouth = face_landmarks[3]
                
                x_mouth = (mouth.x * image.shape[1])
                y_mouth = (mouth.y * image.shape[0])
                
                x_eye_left = (eye_left.x * image.shape[1])
                y_eye_left = (eye_left.y * image.shape[0])
                
                x_eye_right = (eye_right.x * image.shape[1])
                y_eye_right = (eye_right.y * image.shape[0])
                
                x_noise = (noise.x * image.shape[1])
                y_noise = (noise.y * image.shape[0])
                
                if x_mouth > width or y_mouth > height:
                    return False, "Your mouth is not detected! Please show your face"
                    
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
    """Check if face is wearing a mask."""
    x, y, w, h = box
    x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
    # Expand face by 80px 
    x1 = x1 - 80 if x1 - 80 > 0 else 0
    y1 = y1 - 80 if y1 - 80 > 0 else 0
    x2 = x2 + 80 if x2 + 80 < img_decode.shape[1] else img_decode.shape[1]
    y2 = y2 + 80 if y2 + 80 < img_decode.shape[0] else img_decode.shape[0]
    
    face = img_decode[y1:y2, x1:x2]
    face = face.astype('uint8')
    
    try:    
        prediction = model.predict(face)
        class_id = int(prediction[0].boxes[0].cls)

        if class_id == 1 or class_id == 2:
            return False, "Your face is wearing a mask! Please remove the mask"
    except:
        return False, "Please checkin again!"
    return True, "Face is not wearing a mask"


def check_condition(data, is_checkin=True):
    """Check conditions for checkin or registration."""
    if is_checkin == False:
        if data.id is None or data.name is None or data.id == "" or data.name == "":
            return False, "id and name are required"
    
    if len(data.img_base64) == 0:
        return False, "invalid"
    
    if data.role != '1' and data.role != '0':
        return False, "invalid"
    
    if data.store_id is None or data.store_id == "":
        return False, "store_id is required"
    return True, "Success"
