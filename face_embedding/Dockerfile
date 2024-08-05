FROM tensorflow/tensorflow:2.13.0

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN python3 -m pip install --upgrade pip
RUN apt install -y libgl1-mesa-glx
RUN apt install -y libglib2.0-0
RUN pip3 install opencv-python==4.5.*
RUN pip3 install deepface

WORKDIR /face_embedding
COPY . /face_embedding

RUN pip3 install fastapi
RUN pip3 install uvicorn
RUN pip3 install requests
RUN pip3 install python-multipart
# RUN mkdir -p /root/.deepface/weights
# RUN wget -O /root/.deepface/weights/arcface_weights.h5 https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.h5

EXPOSE 2024

# CMD ["python3","-m","uvicorn", "face:app", "--host", "0.0.0.0", "--port", "2024", "--reload"]