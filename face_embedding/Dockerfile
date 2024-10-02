FROM python:3.9

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN python3 -m pip install --upgrade pip
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -y
RUN apt install libgl1-mesa-glx -y
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y
RUN pip3 install opencv-python
RUN pip3 install tensorflow==2.13.0
RUN pip3 install deepface
RUN pip3 install ultralytics
RUN pip3 install mediapipe
RUN pip3 install scipy

WORKDIR /face_embedding
COPY . /face_embedding

RUN pip3 install fastapi
RUN pip3 install uvicorn
RUN pip3 install requests
RUN pip3 install python-multipart

EXPOSE 2024

# CMD ["python3","-m","uvicorn", "face:app", "--host", "0.0.0.0", "--port", "2024", "--reload"]