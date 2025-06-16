FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

# Cài đặt các dependencies
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3.9 \
    python3-pip \
    python3.9-distutils \
    libgl1-mesa-glx \
    ffmpeg \
    libsm6 \
    libxext6 \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Thiết lập Python 3.9 mặc định
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Nâng cấp pip và cài đặt packages
RUN python3 -m pip install --no-cache-dir --upgrade pip

RUN pip3 install opencv-contrib-python==4.10.0.84
RUN pip3 install opencv-python==4.10.0.84
RUN pip3 install 'tensorflow[and-cuda]==2.14.0'
RUN pip3 install --no-cache-dir "pandas==2.2.2"
RUN pip3 install deepface==0.0.93
RUN pip3 install ultralytics==8.2.97
RUN pip3 install mediapipe==0.10.15
RUN pip3 install scipy

RUN pip3 install fastapi
RUN pip3 install uvicorn
RUN pip3 install requests
RUN pip3 install python-multipart
RUN pip3 install python-dotenv
RUN pip3 install boto3
RUN pip3 install httpx
RUN pip3 install aioboto3

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/
RUN pip3 install -r requirements.txt

# Copy source code
COPY main.py /app/
COPY src/ /app/src/
COPY config/ /app/config/
COPY models/ /app/models/
COPY .env /app/

# Create necessary directories
RUN mkdir -p /app/logs /app/static /app/snapshots

# Tải các weights cần thiết
RUN mkdir -p /root/.deepface/weights && \ 
    wget https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5 -O /root/.deepface/weights/vgg_face_weights.h5 && \
    wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb' -O /root/.deepface/weights/yolov8n-face.pt

EXPOSE 8000

# Set the command to run the application
CMD ["python", "main.py"]