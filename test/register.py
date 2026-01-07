import base64
import requests
import os

MY_API_KEY = "gBoyON6XU978cIpI0x1r0Hs0JjL7Ms2cZ0LL27VH6"

image_path = "./image/sontung2.jpg"

with open(image_path, "rb") as f:
    img_bytes = f.read()

img_base64 = base64.b64encode(img_bytes).decode('utf-8')

headers = {
    "Content-Type": "application/json",
    "X-API-Key": MY_API_KEY 
}

try:
    response = requests.post(
        "http://localhost:2024/create_face_img_base64",
        json={
            "img_base64": img_base64,
            "id": "345",
            "name": "Sơn Tùng M-TP",
            "is_update": True,
            "role": "1",
            "store_id": "TEST1"
        },
        headers=headers  
    )

    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())

except Exception as e:
    print("Lỗi khi gọi API:", str(e))