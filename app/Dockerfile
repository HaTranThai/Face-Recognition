from python:3.9

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt --no-cache-dir

EXPOSE 7005
# uvicorn "api:app" -- host 0.0.0.0 -- port 8000 -- reload
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7005", "--reload"]
