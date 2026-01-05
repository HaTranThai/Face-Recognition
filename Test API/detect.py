"""
Script test face detection/recognition
Kiểm tra xem ảnh có detect được face không và nhận diện được ai
"""
import base64
import requests
import sys

# Configuration
FACE_API_URL = "http://localhost:2024"

def test_face_detection(image_path: str, store_id: str, role: str = "1"):
    """
    Test face detection và recognition
    
    Args:
        image_path: Đường dẫn đến file ảnh
        store_id: ID cửa hàng
        role: "1" = Employee, "0" = Customer
    """
    print("=" * 60)
    print("FACE DETECTION TEST")
    print("=" * 60)
    
    # 1. Đọc ảnh
    print(f"\n1. Reading image: {image_path}")
    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        print(f"   ✅ Image size: {len(img_bytes):,} bytes ({len(img_bytes)/1024:.2f} KB)")
    except FileNotFoundError:
        print(f"   ❌ File not found: {image_path}")
        return
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")
        return
    
    # 2. Convert sang base64
    print(f"\n2. Converting to base64...")
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    print(f"   ✅ Base64 length: {len(img_base64):,} characters")
    
    # 3. Test face recognition
    print(f"\n3. Testing face recognition...")
    print(f"   Store ID: {store_id}")
    print(f"   Role: {'Employee' if role == '1' else 'Customer'}")
    
    try:
        response = requests.post(
            f"{FACE_API_URL}/face_recog_img_base64",
            json={
                "img_base64": img_base64,
                "role": role,
                "store_id": store_id
            },
            timeout=60
        )
        
        print(f"\n   📊 Response:")
        print(f"   Status Code: {response.status_code}")
        
        result = response.json()
        status = result.get('status')
        
        if response.status_code == 200:
            if status == 1:
                # Nhận diện thành công
                print(f"   ✅ FACE RECOGNIZED!")
                print(f"   👤 ID: {result.get('id')}")
                print(f"   📝 Name: {result.get('name')}")
            elif status == 0:
                # Không tìm thấy face trong database
                print(f"   ⚠️  FACE NOT FOUND IN DATABASE")
                print(f"   Message: {result.get('message')}")
                print(f"\n   💡 Suggestions:")
                print(f"      - Face chưa được đăng ký")
                print(f"      - Hoặc đăng ký ở store khác")
                print(f"      - Hoặc role khác (employee/customer)")
            else:
                # Lỗi khác
                print(f"   ❌ ERROR")
                print(f"   Message: {result.get('message')}")
        elif response.status_code == 500:
            # Server error - thường do ảnh kém chất lượng
            print(f"   ❌ SERVER ERROR (Image Quality Issue)")
            print(f"   Message: {result.get('message')}")
            print(f"\n   💡 Possible reasons:")
            print(f"      - Face is blurry")
            print(f"      - No face detected in image")
            print(f"      - Eyes are closed")
            print(f"      - Face is not aligned properly")
            print(f"      - Face is too far or too close")
        else:
            print(f"   ⚠️  Unexpected status code: {response.status_code}")
            print(f"   Response: {result}")
            
    except requests.exceptions.Timeout:
        print(f"   ❌ Request timeout (>30s)")
        print(f"   API might be processing a heavy request")
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Cannot connect to API")
        print(f"   Is the service running at {FACE_API_URL}?")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        image_path = sys.argv[1]
        store_id = sys.argv[2] if len(sys.argv) >= 3 else "TEST"
        role = sys.argv[3] if len(sys.argv) >= 4 else "1"
    else:
        image_path = "./image/sontung2.jpg"
        store_id = "TEST1"
        role = "1"
    
    test_face_detection(image_path, store_id, role)
