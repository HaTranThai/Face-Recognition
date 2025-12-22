FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu20.04

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

# -------------------------------------------------------------------
# 1) Install system dependencies
# -------------------------------------------------------------------
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        python3.9 \
        python3-pip \
        python3.9-distutils \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        wget \
    && rm -rf /var/lib/apt/lists/*

# Set python 3.9 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# -------------------------------------------------------------------
# 2) Fix compatibility (TensorFlow 2.13)
# -------------------------------------------------------------------
RUN pip install \
    pydantic==2.5.2 \
    pydantic-settings==2.1.0 \
    typing-extensions==4.9.0

# -------------------------------------------------------------------
# 3) Install Python dependencies
# -------------------------------------------------------------------
RUN pip install \
        tensorflow==2.13.0 \
        opencv-contrib-python==4.10.0.84 \
        opencv-python==4.10.0.84 \
        pandas==2.2.2 \
        deepface==0.0.93 \
        ultralytics==8.2.97 \
        mediapipe==0.10.15 \
        scipy \
        fastapi \
        uvicorn \
        requests \
        python-multipart \
        python-dotenv \
        boto3 \
        aioboto3 \
        aiobotocore \
        httpx \
        Pillow==10.1.0

# -------------------------------------------------------------------
# 4) App files
# -------------------------------------------------------------------
WORKDIR /app

COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY main.py /app/
COPY src/ /app/src/
COPY config/ /app/config/
COPY models/ /app/models/
COPY .env /app/

RUN mkdir -p /app/logs /app/static /app/snapshots

# -------------------------------------------------------------------
# 5) Download weights
# -------------------------------------------------------------------
RUN mkdir -p /root/.deepface/weights && \
    wget -q https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5 \
         -O /root/.deepface/weights/vgg_face_weights.h5 && \
    wget -q --no-check-certificate \
         "https://drive.google.com/uc?export=download&id=1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb" \
         -O /root/.deepface/weights/yolov8n-face.pt

# EXPOSE 2024
# CMD ["python3", "main.py"]
