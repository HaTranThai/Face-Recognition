import base64
import requests

with open("123_Ba_19_03_05.jpg", "rb") as f:
    img_bytes = f.read()

img_base64 = base64.b64encode(img_bytes).decode('utf-8')

response = requests.post(
    "http://localhost:2024/create_face_img_base64",
    json={
        "img_base64": img_base64,
        "id": "123",
        "name": "Ba",
        "role": "1",
        "store_id": "TEST_1"
    }
)

result = response.json()
print(result)