import os
import httpx

SERVER_IP = "127.0.0.1" 
DB_API_PORT = 7005  
SNAPSHOT_DIR = "/home/bbsw/Face-Recognition/app/snapshots"

def process_delete_collections():
    if not os.path.exists(SNAPSHOT_DIR):
        print(f"❌ Thư mục {SNAPSHOT_DIR} không tồn tại!")
        return

    folders = [f for f in os.listdir(SNAPSHOT_DIR) if os.path.isdir(os.path.join(SNAPSHOT_DIR, f))]
    
    if not folders:
        print("ℹ️ Không tìm thấy thư mục nào trong snapshot folder.")
        return

    print(f"🚀 Bắt đầu xóa {len(folders)} collection...")

    for collection_name in folders:
        print(f"\n--- Đang xử lý xóa Collection: {collection_name} ---")
        
        # Endpoint xóa collection đã định nghĩa trong qdrant_database_FE/app.py
        delete_url = f"http://{SERVER_IP}:{DB_API_PORT}/delete_collection"
        
        try:
            # Lưu ý: Endpoint trong app.py sử dụng phương thức DELETE
            # và nhận body qua pydantic model CreateCollection
            with httpx.Client(timeout=30.0) as client:
                response = client.request(
                    "DELETE", 
                    delete_url, 
                    json={"collection_name": collection_name}
                )
                
                if response.status_code == 200:
                    print(f"✅ Đã xóa thành công collection: {collection_name}")
                    print(f"💬 Chi tiết: {response.json().get('message')}")
                elif response.status_code == 404:
                    print(f"⚠️ Collection '{collection_name}' không tồn tại trên database.")
                else:
                    print(f"❌ Lỗi khi xóa: {response.status_code} - {response.text}")
                    
        except Exception as e:
            print(f"❌ Lỗi kết nối khi gọi api_db: {str(e)}")
            continue

if __name__ == "__main__":
    process_delete_collections()